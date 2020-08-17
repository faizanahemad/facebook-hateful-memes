import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
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
    clean_memory
from ..external.detr import get_detr_model, DETRShim
import transformers
import os
import random
import math
from ...utils.ImageModelShims import ImageCaptioningShim, ImageModelShim, ImageModelShimSimple


# t5-small albert-base-v2 ctrl distilgpt2 google/electra-base-generator microsoft/DialoGPT-small allenai/scibert_scivocab_uncased activebus/BERT_Review allenai/reviews_roberta_base
# Hate-speech-CNERG/dehatebert-mono-english nlptown/bert-base-multilingual-uncased-sentiment

#

# allenai/scibert_scivocab_uncased allenai/biomed_roberta_base distilgpt2 google/electra-base-generator ctrl t5-small
# mrm8488/t5-base-finetuned-math-qa-test malteos/arqmath-bert-base-cased

# Keep per head loss stats, make GRU heads separate


class ImageGRUModel(AlbertClassifer):
    def __init__(self, image_model, classifier_dims, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,final_layer_builder,
                 n_tokens_in=640, n_tokens_out=16,
                 use_as_super=False, **kwargs):
        embedding_dims = 768
        super(ImageGRUModel, self).__init__(classifier_dims, num_classes, gaussian_noise, dropout,
                                            internal_dims, n_layers,
                                            "transformer", final_layer_builder,
                                            n_tokens_in, n_tokens_out, True, **kwargs)
        assert n_tokens_in % n_tokens_out == 0
        #
        attention_drop_proba = kwargs.pop("attention_drop_proba", 0.0)
        self.attention_drop_proba = attention_drop_proba

        if type(image_model) == dict:
            module_gaussian = image_model.pop("gaussian_noise", 0.0)
            module_dropout = image_model.pop("dropout", 0.0)
            stored_model = image_model.pop("stored_model", None)
            im_model = ImageModelShimSimple(resnet="resnet18_swsl", dropout=module_dropout, gaussian_noise=module_gaussian, stored_model=stored_model)
        elif type(image_model) == str:
            module_gaussian = 0.0
            module_dropout = 0.0
            stored_model = image_model
            im_model = ImageModelShimSimple(resnet="resnet18_swsl", dropout=module_dropout, gaussian_noise=module_gaussian, stored_model=stored_model)
        elif type(image_model) == ImageModelShimSimple:
            im_model = image_model
        else:
            raise NotImplementedError()

        im_shape = (768, 64)
        if embedding_dims != 768:
            im_proc = nn.Linear(768, embedding_dims)
            init_fc(im_proc, "linear")
            im_proc = [im_proc, nn.Dropout(dropout)]
        else:
            im_proc = []
        im_proc = nn.Sequential(*im_proc, nn.LayerNorm(embedding_dims))
        self.im_model = im_model
        self.post_proc = im_proc
        self.im_shape = im_shape

        self.total_tokens = 1 + 10 + n_tokens_in + 10 + 1
        self.text_tokens = n_tokens_in

        model = kwargs["model"]

        model_class = AutoModel
        tokenizer_class = AutoTokenizer

        global_dir = get_global("models_dir")
        model = os.path.join(global_dir, model) if model in os.listdir(global_dir) else model

        self.tokenizer = tokenizer_class.from_pretrained(model)
        #
        if model == "allenai/scibert_scivocab_uncased":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 768
        elif model == "allenai/biomed_roberta_base":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 768
        elif model == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 768
        elif model == "mrm8488/bert-tiny-5-finetuned-squadv2":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 128
        elif model == "distilgpt2":
            self.word_embeddings = model_class.from_pretrained(model).wte
            self.word_embedding_dims = 768
        elif model == "google/electra-base-generator":
            self.word_embeddings = model_class.from_pretrained(model).embeddings.word_embeddings
            self.word_embedding_dims = 768
        elif model == "ctrl":
            self.word_embeddings = model_class.from_pretrained(model).w
            self.word_embedding_dims = 1280
        elif model == "t5-small":
            self.word_embeddings = model_class.from_pretrained(model).shared
            self.word_embedding_dims = 512
        else:
            raise NotImplementedError()

        fc0 = nn.Linear(self.word_embedding_dims, embedding_dims)
        init_fc(fc0, "leaky_relu")
        self.gru_lin = nn.Sequential(fc0, nn.Dropout(dropout), nn.LeakyReLU(), nn.LayerNorm(embedding_dims))
        self.gru = GRUFeaturizer(self.total_tokens, embedding_dims, n_tokens_out,
                                 classifier_dims, classifier_dims, n_layers, gaussian_noise, dropout)
        #
        print("Pick stored Model", model, "Model Class = ", type(self.model), "Tokenizer Class = ", type(self.tokenizer))
        self.do_mlm = kwargs.pop("do_mlm", False)

        self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)

        self.LayerNorm = nn.LayerNorm(embedding_dims, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])

        self.reg_layers = [(c, c.p if hasattr(c, "p") else c.sigma) for c in self.children() if c.__class__ == GaussianNoise or c.__class__ == nn.Dropout]

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.text_tokens
        if self.training and self.word_masking_proba > 0:
            texts = [random_word_mask(t, tokenizer, self.word_masking_proba) for t in texts]
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def get_vectors(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        img = sampleList.torchvision_image
        input_ids, attention_mask = self.tokenise(sampleList.text)
        word_embeddings = self.word_embeddings(input_ids)
        global_word_view = word_embeddings.mean(1).unsqueeze(1)
        img = img.to(get_device())
        im_repr = self.im_model(img)
        im_repr = self.post_proc(im_repr).to(get_device())
        image_vectors = im_repr[:, :10]
        clean_memory()
        image_vectors = image_vectors.to(get_device())
        embeddings = torch.cat([global_word_view, image_vectors, word_embeddings, image_vectors, global_word_view], 1)

        hidden_state = self.gru(embeddings, not self.do_mlm)
        if self.do_mlm:
            hidden_state = hidden_state[:, 11:11+self.text_tokens]
        return (hidden_state,)

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        labels = torch.tensor(sampleList.label).to(get_device())
        # sample_weights = torch.tensor(sampleList.sample_weight, dtype=float).to(get_device())
        vectors = self.get_vectors(sampleList)[-1]
        logits, loss = self.final_layer(vectors, labels) if self.final_layer is not None else (None, None)

        if self.training:
            loss += self.auc_dice_loss(logits, labels)
        return logits, vectors.mean(1), vectors, loss
