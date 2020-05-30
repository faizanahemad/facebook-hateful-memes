import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
import fasttext
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors
from .FasttextPooled import FasttextPooledModel
from ..classifiers import CNN1DClassifier, GRUClassifier


class Fasttext1DCNNModel(FasttextPooledModel):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0, use_as_submodel=False, embedding_dims=500,
                 internal_dims=512, n_layers=2,
                 classifier="cnn",
                 use_as_super=False,
                 **kwargs):
        super(Fasttext1DCNNModel, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout, use_as_submodel, True, **kwargs)

        if classifier == "cnn":
            self.classifier = CNN1DClassifier(num_classes, 64, 500, 16, 256, internal_dims, None, gaussian_noise, dropout)
        elif classifier == "gru":
            self.classifier = GRUClassifier(num_classes, 64, 500, 16, 256, internal_dims, n_layers, gaussian_noise, dropout)
        else:
            raise NotImplementedError()

    def __get_scores__(self, texts: List[str], img=None):
        pass

    def forward(self, texts: List[str], img, labels):
        vectors = self.get_word_vectors(texts)
        logits, vectors = self.classifier(vectors)
        if self.use_as_submodel:
            loss = None
            preds = None
        else:
            loss = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
            preds = logits.max(dim=1).indices
            logits = torch.softmax(logits, dim=1)

        return logits, preds, vectors.mean(1), vectors, loss

