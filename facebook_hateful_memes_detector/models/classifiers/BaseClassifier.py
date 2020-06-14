from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from ...utils import Transpose, GaussianNoise, init_fc


class NarrowCNNHead(nn.Module):
    pass


class WideCNNHead(nn.Module):
    pass


class AveragedLinearHead(nn.Module):
    pass


class OneTokenPositionLinearHead(nn.Module):
    pass


class WordChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels, strides):
        super(WordChannelReducer, self).__init__()
        self.strides = strides
        conv = nn.Conv1d(in_channels, out_channels * 2, strides, strides, padding=0, groups=4, bias=False)
        init_fc(conv, "leaky_relu")
        conv2 = nn.Conv1d(out_channels * 2, out_channels, 1, 1, padding=0, groups=1, bias=False)
        init_fc(conv2, "linear")
        self.layers = nn.Sequential(Transpose(), conv, nn.LeakyReLU(), conv2, Transpose())

    def forward(self, x):
        assert x.size(-1) % self.strides == 0
        return self.layers(x)


class BaseClassifier(nn.Module):
    def __init__(self, num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):
        super(BaseClassifier, self).__init__()
        assert n_tokens_in % n_tokens_out == 0 and n_tokens_in > n_tokens_out
        self.num_pooling = int(n_tokens_in / n_tokens_out)
        self.n_tokens_out = n_tokens_out
        self.n_channels_out = n_channels_out
        self.n_internal_dims = n_internal_dims
        self.num_classes = num_classes


class BasicFeaturizer(BaseClassifier):
    def __init__(self, num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):
        super(BasicFeaturizer, self).__init__(num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                                              n_internal_dims, n_layers, gaussian_noise, dropout)

        self.features = WordChannelReducer(n_channels_in, n_channels_out, self.num_pooling)

    def forward(self, x):
        return self.features(x)
