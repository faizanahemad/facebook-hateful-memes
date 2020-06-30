import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, BasicFeaturizer
from .Fasttext1DCNN import Fasttext1DCNNModel
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
import torchvision.models as models
from ...utils import get_device


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
        finetune = kwargs["finetune"] if "finetune" in kwargs else False
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.finetune = finetune
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False
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
            else:
                raise NotImplementedError()

            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, )

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        m = lambda x: tokenizer.encode_plus(x, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in)
        input_ids, attention_mask = zip(*[(d['input_ids'], d['attention_mask']) for d in map(m, texts)])
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def get_word_vectors(self, texts: List[str]):
        input_ids, attention_mask = self.tokenise(texts)
        if self.finetune:
            outputs = self.model(input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        pooled_output = outputs[1]
        return last_hidden_states
