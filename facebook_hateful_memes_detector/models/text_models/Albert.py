import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..classifiers import CNN1DClassifier, GRUClassifier
from .Fasttext1DCNN import Fasttext1DCNNModel
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification

import torchvision.models as models


class AlbertClassifer(Fasttext1DCNNModel):
    def __init__(self, classifer_dims, num_classes, embedding_dims,
                 gaussian_noise=0.0, dropout=0.0,
                 internal_dims=512, n_layers=2,
                 classifier="cnn",
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False, **kwargs):
        super(AlbertClassifer, self).__init__(classifer_dims, num_classes, embedding_dims, gaussian_noise, dropout,
                                              internal_dims, n_layers,
                                              classifier,
                                              n_tokens_in, n_tokens_out, True, **kwargs)
        assert n_tokens_in % n_tokens_out == 0
        model = kwargs["model"] if "model" in kwargs else 'albert-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
        self.model = AutoModel.from_pretrained('albert-base-v2')
        for p in self.model.parameters():
            p.requires_grad = False
        if not use_as_super:
            if classifier == "cnn":
                self.classifier = CNN1DClassifier(num_classes, n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifer_dims, internal_dims, None, gaussian_noise, dropout)
            elif classifier == "gru":
                self.classifier = GRUClassifier(num_classes, n_tokens_in, embedding_dims, n_tokens_out, classifer_dims,
                                                internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        m = lambda x: tokenizer.encode_plus(x, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in)
        input_ids, attention_mask = zip(*[(d['input_ids'], d['attention_mask']) for d in map(m, texts)])
        return torch.tensor(input_ids), torch.tensor(attention_mask)

    def get_word_vectors(self, texts: List[str]):
        input_ids, attention_mask = self.tokenise(texts)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        pooled_output = outputs[1]
        return last_hidden_states
