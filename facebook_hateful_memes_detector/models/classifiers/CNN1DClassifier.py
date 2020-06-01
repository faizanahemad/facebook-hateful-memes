from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from .BaseClassifier import BaseClassifier
from ...utils import init_fc, GaussianNoise
import math


# 5 Conv -> R1 -> MP -> R2 -> MP -> R3 -> MP


class Residual1DConv(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False, gaussian_noise=0.0, dropout=0.0):
        super().__init__()
        r1 = nn.Conv1d(in_channels, in_channels * 2, 3, 1, padding=1, groups=4, bias=False)
        init_fc(r1, "leaky_relu")
        r2 = nn.Conv1d(in_channels * 2, in_channels * 2, 3, 1, padding=1, groups=4, bias=False)
        init_fc(r2, "leaky_relu")
        r3 = nn.Conv1d(in_channels * 2, in_channels, 1, 1, padding=0, groups=1, bias=False)
        init_fc(r3, "linear")
        relu = nn.LeakyReLU()
        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        self.r1 = nn.Sequential(dp, r1, relu, gn,)
        self.r2 = nn.Sequential(r2, relu, dp)
        self.r3 = r3

        self.channel_sizer = None
        mul = 1
        if in_channels * mul != out_channels:
            channel_sizer = nn.Conv1d(in_channels * mul, out_channels, 1, 1, padding=0, groups=1, bias=False) # dont change groups here
            init_fc(channel_sizer, "linear")
            self.channel_sizer = nn.Sequential(dp, channel_sizer)

        self.pooling = nn.MaxPool1d(2)
        self.pooling2 = nn.AvgPool1d(2)
        self.pool = pool

    def forward(self, x):
        r1 = self.r1(x)
        r2 = self.r2(r1)
        residual = self.r3(r2)
        x = x + residual

        if self.pool:
            x = self.pooling(x)

        x = self.channel_sizer(x) if self.channel_sizer is not None else x
        return x


class CNN1DClassifier(BaseClassifier):
    def __init__(self, num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):

        super(CNN1DClassifier, self).__init__(num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                                              n_internal_dims, n_layers, gaussian_noise, dropout)
        assert math.log2(self.num_pooling).is_integer()

        l1 = nn.Conv1d(n_channels_in, n_internal_dims, 5, 1, padding=2, groups=4, bias=False)
        init_fc(l1, "leaky_relu")
        l2 = nn.Conv1d(n_internal_dims, n_internal_dims, 3, 1, padding=1, groups=4, bias=False)
        init_fc(l2, "linear")
        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        layers = [l1, nn.LeakyReLU(), gn, l2, dp]
        for _ in range(int(math.log2(self.num_pooling))):
            layers.append(Residual1DConv(n_internal_dims, n_internal_dims, True, gaussian_noise, dropout))
            layers.append(gn)
        layers.append(Residual1DConv(n_internal_dims, n_channels_out, False, gaussian_noise, dropout))
        layers.append(dp)
        self.featurizer = nn.Sequential(*layers)

        self.c1 = nn.Conv1d(n_channels_out, num_classes, 3, 1, padding=0, groups=1, bias=False)
        init_fc(self.c1, "linear")
        self.avp = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.featurizer(x)
        logits = self.avp(self.c1(x)).squeeze()
        x = x.transpose(1, 2)
        assert x.size(1) == self.n_tokens_out and x.size(2) == self.n_channels_out
        return logits, x







