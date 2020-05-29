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
from .WordChannelReducer import Squeeze, Transpose


class Fasttext1DCNNModel(FasttextPooledModel):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0, use_as_submodel=False, embedding_dims=500, cnn_dims=512, use_as_super=False,
                 **kwargs):
        super(Fasttext1DCNNModel, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout, use_as_submodel, True, **kwargs)

        conv1 = nn.Conv1d(embedding_dims, cnn_dims, 3, 1, padding=1, groups=4, bias=False)
        init_fc(conv1, "leaky_relu")
        mp = nn.MaxPool1d(2)
        conv2 = nn.Conv1d(cnn_dims, cnn_dims * 2, 3, 1, padding=1, groups=4, bias=False)
        init_fc(conv2, "leaky_relu")
        conv3 = nn.Conv1d(cnn_dims * 2, cnn_dims, 3, 1, padding=1, groups=4, bias=False)
        init_fc(conv3, "leaky_relu")
        relu = nn.LeakyReLU()
        dropout = nn.Dropout(dropout)
        gn = GaussianNoise(gaussian_noise)
        self.conv = nn.Sequential(gn, Transpose(), conv1, relu, dropout, mp,
                                  conv2, relu, dropout, mp,
                                  conv3, relu, dropout, mp,
                                  Transpose())
        self.classifier = nn.Sequential(Transpose(), nn.Conv1d(cnn_dims, 2, 8, 1, padding=0, groups=1, bias=False), Squeeze())
        # init_fc(self.lstm, 'linear')

    def __get_scores__(self, texts: List[str], img=None):
        vectors = self.get_word_vectors(texts)
        conv_out = self.conv(vectors)
        mean_projection = conv_out.mean(1)
        return mean_projection, conv_out

    def forward(self, texts: List[str], img, labels):
        projections, vectors = self.__get_scores__(texts, img)
        if self.use_as_submodel:
            loss = None
            preds = None
            logits = None
        else:
            logits = self.classifier(vectors) if not self.use_as_submodel else None
            loss = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
            preds = logits.max(dim=1).indices
            logits = torch.softmax(logits, dim=1)

        return logits, preds, projections, vectors, loss

