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


class DualWideCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, pool=False, gaussian_noise=0.0, dropout=0.0):
        super().__init__()
        assert activation in ["linear", "leaky_relu"]
        r1 = nn.Conv1d(in_channels, in_channels * 2, 3, 1, padding=1, groups=4, bias=False)
        r2 = nn.Conv1d(in_channels * 2, out_channels, 3, 1, padding=1, groups=1, bias=False)
        init_fc(r1, "leaky_relu")
        init_fc(r2, activation)
        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        layers = [gn, r1, dp, nn.LeakyReLU(), gn, r2, dp]
        if activation != "linear":
            layers.append(nn.LeakyReLU())

        self.cnn = nn.Sequential(*layers)
        self.pooling = nn.MaxPool1d(2)
        self.pool = pool


    def forward(self, x):
        x = self.cnn(x)
        if self.pool:
            x = self.pooling(x)
        return x


class DualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, pool=False, gaussian_noise=0.0, dropout=0.0):
        super().__init__()
        assert activation in ["linear", "leaky_relu"]
        r1 = nn.Conv1d(in_channels, in_channels, 3, 1, padding=1, groups=8, bias=False)
        r2 = nn.Conv1d(in_channels, out_channels, 3, 1, padding=1, groups=1, bias=False)
        init_fc(r1, "leaky_relu")
        init_fc(r2, activation)
        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        layers = [gn, r1, dp, nn.LeakyReLU(), gn, r2, dp, ]
        if activation != "linear":
            layers.append(nn.LeakyReLU())

        self.cnn = nn.Sequential(*layers)
        self.pooling = nn.MaxPool1d(2)
        self.pool = pool

    def forward(self, x):
        x = self.cnn(x)
        if self.pool:
            x = self.pooling(x)
        return x


class CNN1DSimple(BaseClassifier):
    def __init__(self, num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):

        super(CNN1DSimple, self).__init__(num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                                              n_internal_dims, n_layers, gaussian_noise, dropout)
        assert math.log2(self.num_pooling).is_integer()

        l1 = nn.Conv1d(n_channels_in, n_internal_dims, 5, 1, padding=2, groups=1, bias=False)
        init_fc(l1, "leaky_relu")

        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        layers = [dp, l1, nn.LeakyReLU(), gn]
        for _ in range(int(math.log2(self.num_pooling))):
            layers.append(DualCNNBlock(n_internal_dims, n_internal_dims, "leaky_relu", False, gaussian_noise, dropout))
            layers.append(DualWideCNNBlock(n_internal_dims, n_internal_dims, "leaky_relu", True, gaussian_noise, dropout))
        layers.append(DualCNNBlock(n_internal_dims, n_channels_out, "linear", False, gaussian_noise, dropout))
        self.featurizer = nn.Sequential(*layers)

        c1 = nn.Conv1d(n_channels_out, num_classes, 3, 1, padding=0, groups=1, bias=False)
        init_fc(c1, "linear")
        avp = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(c1, avp) # Residual1DConv(n_channels_out, n_channels_out, True, gaussian_noise, dropout),

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.featurizer(x)
        logits = self.classifier(x).squeeze()
        x = x.transpose(1, 2)
        assert x.size(1) == self.n_tokens_out and x.size(2) == self.n_channels_out
        return logits, x







