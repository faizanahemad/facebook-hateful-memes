import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, BasicFeaturizer, TransformerFeaturizer
from .Fasttext1DCNN import Fasttext1DCNNModel
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
import torchvision.models as models
from ...utils import get_device, GaussianNoise, random_word_mask
import random


class AlbertClassifer(Fasttext1DCNNModel):
    def __init__(self, classifier_dims, num_classes, embedding_dims,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False, **kwargs):
        super(AlbertClassifer, self).__init__(classifier_dims, num_classes, embedding_dims, gaussian_noise, dropout,
                                              internal_dims, n_layers,
                                              featurizer, final_layer_builder,
                                              n_tokens_in, n_tokens_out, True, **kwargs)
        assert n_tokens_in % n_tokens_out == 0
        model = kwargs["model"] if "model" in kwargs else 'albert-base-v2'
        self.word_masking_proba = kwargs["word_masking_proba"] if "word_masking_proba" in kwargs else 0.0
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        if not use_as_super:
            if featurizer == "cnn":
                self.featurizer = CNN1DFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifier_dims, internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "gru":
                self.featurizer = GRUFeaturizer(n_tokens_in, embedding_dims, n_tokens_out, classifier_dims,
                                                internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "basic":
                self.featurizer = BasicFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifier_dims,
                                                  internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "transformer":
                n_encoders = kwargs["n_encoders"] if "n_encoders" in kwargs else n_layers
                n_decoders = kwargs["n_decoders"] if "n_decoders" in kwargs else n_layers
                self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                        classifier_dims,
                                                        internal_dims, n_encoders, n_decoders, gaussian_noise, dropout)
            else:
                raise NotImplementedError()

            loss = kwargs["loss"] if "loss" in kwargs else None
            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, loss)

        self.reg_layers = [(c, c.p if hasattr(c, "p") else c.sigma) for c in self.children() if c.__class__ == GaussianNoise or c.__class__ == nn.Dropout]

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        if self.training and self.word_masking_proba > 0:
            texts = [random_word_mask(t, tokenizer, self.word_masking_proba) for t in texts]
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def get_word_vectors(self, texts: List[str]):
        input_ids, attention_mask = self.tokenise(texts)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        pooled_output = outputs[1]
        return last_hidden_states
