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
        lstm = nn.GRU(n_channels_in, int(n_internal_dims/2), n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.ln = nn.LayerNorm(n_internal_dims)
        if n_internal_dims != n_channels_out:
            projection = nn.Linear(n_internal_dims, n_channels_out)
            init_fc(projection, "leaky_relu")
            self.projection = nn.Sequential(GaussianNoise(gaussian_noise), projection, nn.LeakyReLU())
        self.indices = list(reversed(range(n_tokens_in-1, 0, -self.num_pooling)))[:self.n_tokens_out]
        self.featurizer = lstm

    def forward(self, x, filter_indices=True):
        x, _ = self.featurizer(x)
        x = self.ln(x)
        if filter_indices:
            x = x[:, self.indices]
            assert x.size(1) == self.n_tokens_out
        if hasattr(self, "projection"):
            x = self.projection(x)
        x.size(2) == self.n_channels_out
        return x


