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

class TransformerImageV2Model(nn.Module):
    def __init__(self, image_model, classifier_dims, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=64, n_tokens_out=16,
                 head_masks=0,
                 use_as_super=False, **kwargs):
        embedding_dims = 768
        super(TransformerImageV2Model, self).__init__()
        #
        self.head_masks = head_masks
        assert self.head_masks <= 12
        attention_drop_proba = kwargs.pop("attention_drop_proba", 0.0)
        self.attention_drop_proba = attention_drop_proba
        numbers_dim = kwargs.pop("numbers_dim", False)
        image_dim = kwargs.pop("image_dim", False)
        embed1_dim = kwargs.pop("embed1_dim", False)
        embed2_dim = kwargs.pop("embed2_dim", False)

        def expand(x):
            return x.unsqueeze(1)

        self.feature_dropout = FeatureDropout(dropout)
        if numbers_dim:
            fc0 = nn.Linear(numbers_dim, 512)
            init_fc(fc0, "leaky_relu")
            fc1 = nn.Linear(512, embedding_dims)
            init_fc(fc1, "leaky_relu")
            self.numbers_embed = nn.Sequential(fc0, nn.LeakyReLU(), nn.Dropout(dropout), fc1, nn.LeakyReLU(), GaussianNoise(gaussian_noise),
                                               LambdaLayer(expand))
            self.NumberLayerNorm = nn.LayerNorm(embedding_dims, eps=1e-12)

        if embed1_dim:
            fc0 = nn.Linear(embed1_dim, min(embed1_dim * 4, 1024))
            init_fc(fc0, "leaky_relu")
            fc1 = nn.Linear(min(embed1_dim * 4, 1024), embedding_dims)
            init_fc(fc1, "leaky_relu")
            self.embed1_embed = nn.Sequential(fc0, nn.LeakyReLU(), nn.Dropout(dropout), fc1, nn.LeakyReLU(), GaussianNoise(gaussian_noise),
                                               LambdaLayer(expand))
            self.Embed1LayerNorm = nn.LayerNorm(embedding_dims, eps=1e-12)

        if embed2_dim:
            fc0 = nn.Linear(embed2_dim, min(embed2_dim * 4, 1024))
            init_fc(fc0, "leaky_relu")
            fc1 = nn.Linear(min(embed2_dim * 4, 1024), embedding_dims)
            init_fc(fc1, "leaky_relu")
            self.embed2_embed = nn.Sequential(fc0, nn.LeakyReLU(), nn.Dropout(dropout), fc1, nn.LeakyReLU(), GaussianNoise(gaussian_noise),
                                               LambdaLayer(expand))
            self.Embed2LayerNorm = nn.LayerNorm(embedding_dims, eps=1e-12)

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
                im_proc = [im_proc, nn.Dropout(dropout)]
            else:
                im_proc = [nn.Identity()]

            im_proc.append(LambdaLayer(expand))
            im_proc = nn.Sequential(*im_proc)
            self.im_model = im_model
            self.post_proc = im_proc
            self.LayerNorm = nn.LayerNorm(embedding_dims, eps=1e-12)

        self.total_tokens = n_tokens_in + int(bool(image_dim)) + int(bool(numbers_dim)) + int(bool(embed1_dim)) + int(bool(embed2_dim))
        self.text_tokens = n_tokens_in

        model = kwargs["model"]
        model_class = AutoModel
        tokenizer_class = AutoTokenizer

        global_dir = get_global("models_dir")
        model = os.path.join(global_dir, model) if model in os.listdir(global_dir) else model
        self.tokenizer = tokenizer_class.from_pretrained(model)
        self.model = model_class.from_pretrained(model)
        print("Pick stored Model", model, "Model Class = ", type(self.model), "Tokenizer Class = ", type(self.tokenizer))
        if featurizer == "transformer":
            n_encoders = kwargs.pop("n_encoders", n_layers)
            n_decoders = kwargs.pop("n_decoders", n_layers)
            self.featurizer = TransformerFeaturizer(self.total_tokens, embedding_dims, n_tokens_out,
                                                    classifier_dims,
                                                    internal_dims, n_encoders, n_decoders,
                                                    gaussian_noise, dropout, self.attention_drop_proba)
        else:
            raise NotImplementedError()

        self.do_mlm = kwargs.pop("do_mlm", False)
        self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)
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
        input_ids, attention_mask = self.tokenise(sampleList.text)
        word_embeddings = self.model.embeddings(input_ids)  # B, S, C
        embeddings = word_embeddings
        if hasattr(sampleList, "torchvision_image"):
            img = sampleList.torchvision_image
            img = img.to(get_device())
            im_repr = self.im_model(img)
            image_vectors = self.post_proc(im_repr).to(get_device())
            clean_memory()
            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.size(0), image_vectors.size(1), device=get_device(), dtype=attention_mask.dtype)],
                1)

            seq_length = embeddings.size(1)
            position_ids = torch.arange(seq_length, seq_length + image_vectors.size(1), dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand(image_vectors.size()[:2])  # (bs, max_seq_length)
            position_embeddings = self.model.embeddings.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

            image_vectors = image_vectors + position_embeddings  # (bs, max_seq_length, dim) # * sqrt(dim)
            image_vectors = self.LayerNorm(image_vectors)  # (bs, max_seq_length, dim)
            image_vectors = image_vectors.to(get_device())
            embeddings = torch.cat([embeddings, image_vectors], 1)

        if hasattr(sampleList, "numbers"):
            numbers = sampleList.numbers
            numbers = numbers.to(get_device())
            numbers = self.feature_dropout(numbers)
            numbers = self.numbers_embed(numbers)
            clean_memory()
            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.size(0), 1, device=get_device(), dtype=attention_mask.dtype)],
                1)

            seq_length = embeddings.size(1)
            position_ids = torch.arange(seq_length, seq_length + numbers.size(1), dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand(numbers.size()[:2])  # (bs, max_seq_length)
            position_embeddings = self.model.embeddings.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
            numbers = numbers + position_embeddings  # (bs, max_seq_length, dim) # * sqrt(dim)
            numbers = self.NumberLayerNorm(numbers)  # (bs, max_seq_length, dim)
            numbers = numbers.to(get_device())
            embeddings = torch.cat([embeddings, numbers], 1)


        if hasattr(sampleList, "embed1"):
            embed1 = sampleList.embed1
            embed1 = embed1.to(get_device())
            embed1 = self.feature_dropout(embed1)
            embed1 = self.embed1_embed(embed1)
            clean_memory()
            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.size(0), 1, device=get_device(), dtype=attention_mask.dtype)],
                1)


            seq_length = embeddings.size(1)
            position_ids = torch.arange(seq_length, seq_length + embed1.size(1), dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand(embed1.size()[:2])  # (bs, max_seq_length)
            position_embeddings = self.model.embeddings.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
            embed1 = embed1 + position_embeddings  # (bs, max_seq_length, dim) # * sqrt(dim)
            embed1 = self.Embed1LayerNorm(embed1)  # (bs, max_seq_length, dim)
            embed1 = embed1.to(get_device())
            embeddings = torch.cat([embeddings, embed1], 1)


        if hasattr(sampleList, "embed2"):
            embed2 = sampleList.embed2
            embed2 = embed2.to(get_device())
            embed2 = self.feature_dropout(embed2)
            embed2 = self.embed2_embed(embed2)
            clean_memory()
            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.size(0), 1, device=get_device(), dtype=attention_mask.dtype)],
                1)


            seq_length = embeddings.size(1)
            position_ids = torch.arange(seq_length, seq_length + embed2.size(1), dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand(embed2.size()[:2])  # (bs, max_seq_length)
            position_embeddings = self.model.embeddings.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
            embed2 = embed2 + position_embeddings  # (bs, max_seq_length, dim) # * sqrt(dim)
            embed2 = self.Embed2LayerNorm(embed2)  # (bs, max_seq_length, dim)
            embed2 = embed2.to(get_device())
            embeddings = torch.cat([embeddings, embed2], 1)

        attention_mask = attention_mask.to(get_device())

        if self.training:
            head_mask = [1] * (12 - self.head_masks) + [0] * self.head_masks
            random.shuffle(head_mask)
        else:
            head_mask = [1] * 12
        encoder = getattr(self.model, "transformer", getattr(self.model, "encoder", None))

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        tfmr_output = encoder(embeddings, attention_mask, head_mask=head_mask)
        hidden_state = tfmr_output[0]
        hidden_state = self.featurizer.forward(hidden_state, filter_indices=not self.do_mlm)
        if self.do_mlm:
            hidden_state = hidden_state[:, :self.text_tokens]
        return (hidden_state,)

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        labels = torch.tensor(sampleList.label).to(get_device())
        # sample_weights = torch.tensor(sampleList.sample_weight, dtype=float).to(get_device())
        vectors = self.get_vectors(sampleList)[-1]
        logits, loss = None, None
        if not self.do_mlm:
            logits, loss = self.final_layer(vectors, labels) if self.final_layer is not None else (None, None)
            if self.training:
                loss += self.auc_dice_loss(logits, labels)
        return logits, vectors.mean(1), vectors, loss
