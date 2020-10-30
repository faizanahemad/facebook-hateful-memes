import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, BasicFeaturizer, TransformerFeaturizer
from .Fasttext1DCNN import Fasttext1DCNNModel
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel, DistilBertTokenizer, LongformerTokenizer
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification, DistilBertModel, LongformerModel
import torchvision.models as models
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb
from ...utils import get_device, GaussianNoise, random_word_mask, load_stored_params, ExpandContract, Transformer, PositionalEncoding, LambdaLayer, get_global, \
    get_regularization_layers, WordMasking
from ...training import fb_1d_loss_builder
import os
import random
import math


class BERTClassifier(nn.Module):
    def __init__(self, classifier_dims, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 device,
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False, **kwargs):
        embedding_dims = 768
        super(BERTClassifier, self).__init__()
        self.word_masking_proba = kwargs["word_masking_proba"] if "word_masking_proba" in kwargs else 0.0

        if not use_as_super:
            model = kwargs["model"] if "model" in kwargs else 'albert-base-v2'
            global_dir = get_global("models_dir")
            model = os.path.join(global_dir, model) if model in os.listdir(global_dir) else model
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModel.from_pretrained(model)
            print("Pick stored Model", model, "Model Class = ", type(self.model), "Tokenizer Class = ", type(self.tokenizer))

            self.attention_drop_proba = kwargs["attention_drop_proba"] if "attention_drop_proba" in kwargs else 0.0
            n_encoders = kwargs.pop("n_encoders", n_layers)
            n_decoders = kwargs.pop("n_decoders", n_layers)
            self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                    classifier_dims,
                                                    internal_dims, n_encoders, n_decoders,
                                                    gaussian_noise, dropout, self.attention_drop_proba)

            self.final_layer = fb_1d_loss_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)
        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.word_masking = WordMasking(tokenizer=self.tokenizer, **kwargs)
        self.device=device

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        texts = self.word_masking(texts)
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(self.device), torch.tensor(attention_mask).to(self.device)

    def get_word_vectors(self, texts: List[str]):
        input_ids, attention_mask = self.tokenise(texts)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        pooled_output = outputs[1]
        return last_hidden_states

    def forward(self, texts: List[str], labels: List[int] = None):
        if labels is not None:
            labels = torch.tensor(labels).to(self.device)
        vectors = self.get_word_vectors(texts)
        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels) if self.final_layer is not None else (None, None)
        return logits, vectors.mean(1), vectors, loss
