from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from .BaseFeaturizer import BaseFeaturizer
from ...utils import Transpose, GaussianNoise, init_fc


class GRUFeaturizer(BaseFeaturizer):
    def __init__(self, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):
        super(GRUFeaturizer, self).__init__(n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                                            n_internal_dims, n_layers, gaussian_noise, dropout)

        assert n_internal_dims % 2 == 0
        lstm = nn.Sequential(GaussianNoise(gaussian_noise),
            nn.GRU(n_channels_in, int(n_internal_dims/2), n_layers, batch_first=True, bidirectional=True, dropout=dropout))

        # conv = nn.Conv1d(n_internal_dims, n_channels_out, 1, 1, padding=0, groups=4, bias=False)
        # init_fc(conv, "linear")
        # pool = nn.AvgPool1d(self.num_pooling)
        # self.projection = nn.Sequential(Transpose(), conv, pool, Transpose())
        projection = nn.Linear(n_internal_dims, n_channels_out)
        init_fc(projection, "leaky_relu")
        self.indices = list(reversed(range(n_tokens_in-1, 0, -self.num_pooling)))
        self.projection = nn.Sequential(projection, nn.LeakyReLU(), nn.LayerNorm(n_channels_out))
        self.featurizer = lstm

    def forward(self, x):
        x, _ = self.featurizer(x)
        x = x[:, self.indices]
        x = self.projection(x)
        assert x.size(1) == self.n_tokens_out and x.size(2) == self.n_channels_out
        return x


