from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from ...utils import Transpose, GaussianNoise, init_fc, Average, WordChannelReducer


class BaseFeaturizer(nn.Module):
    def __init__(self, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):
        super(BaseFeaturizer, self).__init__()
        assert n_tokens_in % n_tokens_out == 0 and n_tokens_in > n_tokens_out
        self.num_pooling = int(n_tokens_in / n_tokens_out)
        self.n_tokens_out = n_tokens_out
        self.n_channels_out = n_channels_out
        self.n_internal_dims = n_internal_dims


class PassThroughFeaturizer(nn.Module):
    def __init__(self, num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):
        super().__init__()
        assert n_tokens_in == n_tokens_out
        assert n_channels_in == n_channels_out
        self.n_tokens_in = n_tokens_in
        self.n_tokens_out = n_tokens_out
        self.n_channels_out = n_channels_out
        self.n_internal_dims = n_internal_dims

    def forward(self, x):
        assert x.size(-1) == self.n_channels_out
        assert x.size(1) == self.n_tokens_out
        return x


class BasicFeaturizer(BaseFeaturizer):
    def __init__(self, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):
        super(BasicFeaturizer, self).__init__(n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                                              n_internal_dims, n_layers, gaussian_noise, dropout)

        self.features = WordChannelReducer(n_channels_in, n_channels_out, self.num_pooling)

    def forward(self, x):
        return self.features(x)
