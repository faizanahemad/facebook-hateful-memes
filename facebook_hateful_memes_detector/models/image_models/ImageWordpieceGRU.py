import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from ...training import get_auc_dice_loss
from ...utils.sample import SampleList, Sample

from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, BasicFeaturizer, TransformerFeaturizer
from ..text_models import AlbertClassifer
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel, LongformerTokenizer, LongformerModel, DistilBertTokenizer, DistilBertModel
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
import torchvision.models as models
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb
from ...utils import get_device, GaussianNoise, random_word_mask, load_stored_params, ExpandContract, Transformer, PositionalEncoding, LambdaLayer, get_global, \
    get_torchvision_classification_models, get_image_info_fn, LambdaLayer, get_vgg_face_model, PositionalEncoding2D, Transpose, init_fc, dict2sampleList, \
    clean_memory, get_regularization_layers, WordMasking, FeatureDropout
from ..external.detr import get_detr_model, DETRShim
import transformers
import os
import random
import math
from ...utils.ImageModelShims import ImageCaptioningShim, ImageModelShim, ImageModelShimSimple, get_shim_resnet


# t5-small albert-base-v2 ctrl distilgpt2 google/electra-base-generator microsoft/DialoGPT-small allenai/scibert_scivocab_uncased activebus/BERT_Review allenai/reviews_roberta_base
# Hate-speech-CNERG/dehatebert-mono-english nlptown/bert-base-multilingual-uncased-sentiment

#

# allenai/scibert_scivocab_uncased allenai/biomed_roberta_base distilgpt2 google/electra-base-generator ctrl t5-small
# mrm8488/t5-base-finetuned-math-qa-test malteos/arqmath-bert-base-cased

# Keep per head loss stats, make GRU heads separate


class ImageGRUModel(nn.Module):
    def __init__(self, image_model, classifier_dims, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, n_layers, final_layer_builder,
                 n_tokens_in=640, n_tokens_out=16,
                 use_as_super=False, **kwargs):
        embedding_dims = internal_dims
        super(ImageGRUModel, self).__init__()
        assert n_tokens_in % n_tokens_out == 0
        #
        attention_drop_proba = kwargs.pop("attention_drop_proba", 0.0)
        self.attention_drop_proba = attention_drop_proba

        numbers_dim = kwargs.pop("numbers_dim", False)
        image_dim = kwargs.pop("image_dim", False)
        embed1_dim = kwargs.pop("embed1_dim", False)
        embed2_dim = kwargs.pop("embed1_dim", False)

        def expand(x):
            return x.unsqueeze(1)

        self.feature_dropout = FeatureDropout(dropout)
        if numbers_dim:
            fc0 = nn.Linear(numbers_dim, 512)
            init_fc(fc0, "leaky_relu")
            fc1 = nn.Linear(512, embedding_dims)
            init_fc(fc1, "leaky_relu")
            self.numbers_embed = nn.Sequential(fc0, nn.LeakyReLU(), nn.Dropout(dropout), fc1, nn.LeakyReLU(), GaussianNoise(gaussian_noise), nn.LayerNorm(embedding_dims))

        if embed1_dim:
            fc0 = nn.Linear(embed1_dim, min(embed1_dim * 4, 1024))
            init_fc(fc0, "leaky_relu")
            fc1 = nn.Linear(min(embed1_dim * 4, 1024), embedding_dims)
            init_fc(fc1, "leaky_relu")
            self.embed1_embed = nn.Sequential(fc0, nn.LeakyReLU(), nn.Dropout(dropout), fc1, nn.LeakyReLU(), GaussianNoise(gaussian_noise),
                                               nn.LayerNorm(embedding_dims))

        if embed2_dim:
            fc0 = nn.Linear(embed2_dim, min(embed2_dim * 4, 1024))
            init_fc(fc0, "leaky_relu")
            fc1 = nn.Linear(min(embed2_dim * 4, 1024), embedding_dims)
            init_fc(fc1, "leaky_relu")
            self.embed2_embed = nn.Sequential(fc0, nn.LeakyReLU(), nn.Dropout(dropout), fc1, nn.LeakyReLU(), GaussianNoise(gaussian_noise),
                                               nn.LayerNorm(embedding_dims))

        if image_dim:
            if type(image_model) == dict:
                module_dropout = image_model.pop("dropout", 0.0)
                stored_model = image_model.pop("stored_model", None)
                im_model = get_shim_resnet(resnet='resnet18_swsl', dropout=module_dropout, dims=image_dim, stored_model=stored_model)
            elif type(image_model) == str:
                module_dropout = 0.0
                stored_model = image_model
                im_model = get_shim_resnet(resnet='resnet18_swsl', dropout=module_dropout, dims=image_dim, stored_model=stored_model)
            else:
                raise NotImplementedError()

            if embedding_dims != image_dim:
                im_proc = nn.Linear(image_dim, embedding_dims)  # TODO: Try conv1D grouped, less params
                init_fc(im_proc, "linear")
                im_proc = [im_proc, nn.Identity()]
            else:
                im_proc = []

            im_proc.append(LambdaLayer(expand))
            im_proc = nn.Sequential(*im_proc, nn.LayerNorm(embedding_dims))
            self.im_model = im_model
            self.post_proc = im_proc

        self.text_tokens = n_tokens_in
        self.skips = int(bool(image_dim)) + int(bool(numbers_dim)) + int(bool(embed1_dim)) + int(bool(embed2_dim))
        self.total_tokens = self.skips + n_tokens_in + self.skips

        model = kwargs["model"]

        model_class = AutoModel

        global_dir = get_global("models_dir")
        model = os.path.join(global_dir, model) if model in os.listdir(global_dir) else model

        tokenizer = AutoTokenizer.from_pretrained(model)
        if model in ["t5-small", "distilgpt2"]:
            setattr(tokenizer, "mask_token_id", tokenizer.pad_token_id)
            setattr(tokenizer, "mask_token", tokenizer.pad_token)
        self.tokenizer = tokenizer
        if model == "allenai/scibert_scivocab_uncased":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 768
        elif model == "allenai/biomed_roberta_base":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 768
        elif model == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 768
        elif model == "ctrl":
            self.word_embeddings = model_class.from_pretrained(model).w
            self.word_embedding_dims = 1280
        #
        elif model == "mrm8488/bert-tiny-5-finetuned-squadv2":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 128
        elif model == "distilgpt2":
            self.word_embeddings = model_class.from_pretrained(model).wte
            self.word_embedding_dims = 768
        elif model == "google/electra-base-generator":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 768
        elif model == "t5-small":
            self.word_embeddings = model_class.from_pretrained(model).shared
            self.word_embedding_dims = 512
        else:
            raise NotImplementedError()

        if self.word_embedding_dims != embedding_dims:
            fc0 = nn.Linear(self.word_embedding_dims, embedding_dims)
            init_fc(fc0, "leaky_relu")
            self.gru_lin = nn.Sequential(fc0, nn.Dropout(dropout), nn.LeakyReLU(), nn.LayerNorm(embedding_dims))
        else:
            self.gru_lin = nn.LayerNorm(embedding_dims)
        self.gru = GRUFeaturizer(self.total_tokens, embedding_dims, n_tokens_out,
                                 classifier_dims, classifier_dims, n_layers, gaussian_noise, dropout)

        self.do_mlm = kwargs.pop("do_mlm", False)
        self.gru_out = nn.LayerNorm(embedding_dims)

        self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)

        self.LayerNorm = nn.LayerNorm(embedding_dims, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.word_masking = WordMasking(tokenizer=self.tokenizer, **kwargs)
        self.reg_layers = get_regularization_layers(self)
        self.auc_loss_coef = kwargs.pop("auc_loss_coef", 0.0)
        self.dice_loss_coef = kwargs.pop("dice_loss_coef", 0.0)
        self.auc_method = kwargs.pop("auc_method", 1)
        self.auc_dice_loss = get_auc_dice_loss(num_classes, self.dice_loss_coef, self.auc_loss_coef, auc_method=self.auc_method)

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.text_tokens
        texts = self.word_masking(texts)
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def get_vectors(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        input_ids, _ = self.tokenise(sampleList.text)
        word_embeddings = self.gru_lin(self.word_embeddings(input_ids))
        word_embeddings = self.feature_dropout(word_embeddings)
        embeddings = word_embeddings
        if hasattr(sampleList, "torchvision_image"):
            img = sampleList.torchvision_image
            img = img.to(get_device())
            im_repr = self.im_model(img)
            image_vectors = self.post_proc(im_repr).to(get_device())
            image_vectors = self.feature_dropout(image_vectors)
            image_vectors = image_vectors.to(get_device())
            embeddings = torch.cat([image_vectors, embeddings, image_vectors], 1)

        if hasattr(sampleList, "numbers"):
            numbers = sampleList.numbers
            numbers = numbers.to(get_device())
            numbers = self.feature_dropout(numbers)
            numbers = self.numbers_embed(numbers)
            embeddings = torch.cat([numbers, embeddings, numbers], 1)

        if hasattr(sampleList, "embed1"):
            embed1 = sampleList.embed1
            embed1 = embed1.to(get_device())
            embed1 = self.feature_dropout(embed1)
            embed1 = self.embed1_embed(embed1)
            embeddings = torch.cat([embed1, embeddings, embed1], 1)

        if hasattr(sampleList, "embed2"):
            embed2 = sampleList.embed2
            embed2 = embed2.to(get_device())
            embed2 = self.feature_dropout(embed2)
            embed2 = self.embed2_embed(embed2)
            embeddings = torch.cat([embed2, embeddings, embed2], 1)
        
        hidden_state = self.gru_out(self.gru.forward(embeddings, filter_indices=not self.do_mlm))
        if self.do_mlm:
            hidden_state = hidden_state[:, self.skips:self.skips+self.text_tokens]
        return (hidden_state,)

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        labels = torch.tensor(sampleList.label).to(get_device())
        # sample_weights = torch.tensor(sampleList.sample_weight, dtype=float).to(get_device())
        vectors = self.get_vectors(sampleList)[-1]
        logits, loss = None, None
        if not self.do_mlm:
            if self.final_layer is not None:
                logits, loss = self.final_layer(vectors, labels)
                if self.training:
                    loss += self.auc_dice_loss(logits, labels)
        return logits, vectors.mean(1), vectors, loss
