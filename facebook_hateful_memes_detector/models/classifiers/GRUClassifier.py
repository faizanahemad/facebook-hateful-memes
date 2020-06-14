from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from .BaseClassifier import BaseClassifier, WordChannelReducer
from ...utils import Transpose, GaussianNoise, init_fc


class GRUClassifier(BaseClassifier):
    def __init__(self, num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):
        super(GRUClassifier, self).__init__(num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                                            n_internal_dims, n_layers, gaussian_noise, dropout)

        assert n_internal_dims % 2 == 0
        lstm = nn.Sequential(GaussianNoise(gaussian_noise),
            nn.GRU(n_channels_in, int(n_internal_dims/2), n_layers, batch_first=True, bidirectional=True, dropout=dropout))

        # conv = nn.Conv1d(n_internal_dims, n_channels_out, 1, 1, padding=0, groups=4, bias=False)
        # init_fc(conv, "linear")
        # pool = nn.AvgPool1d(self.num_pooling)
        # self.projection = nn.Sequential(Transpose(), conv, pool, Transpose())
        projection = WordChannelReducer(n_internal_dims, n_channels_out, self.num_pooling)
        self.projection = nn.Sequential(GaussianNoise(gaussian_noise), projection)
        self.featurizer = lstm
        self.c1 = nn.Conv1d(n_channels_out, num_classes, n_tokens_out, 1, padding=0, groups=1, bias=False)
        init_fc(self.c1, "linear")
        self.avp = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x, _ = self.featurizer(x)
        x = self.projection(x)
        logits = self.avp(self.c1(x.transpose(1, 2))).squeeze()
        assert x.size(1) == self.n_tokens_out and x.size(2) == self.n_channels_out
        return logits, x


