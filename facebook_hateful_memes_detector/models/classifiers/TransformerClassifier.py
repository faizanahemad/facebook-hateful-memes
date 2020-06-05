from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from .BaseClassifier import BaseClassifier
from ...utils import init_fc, GaussianNoise
import math
from .CNN1DClassifier import Residual1DConv


class TransformerClassifier(BaseClassifier):
    def __init__(self, num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_layers,
                 gaussian_noise=0.0, dropout=0.0):
        super(TransformerClassifier, self).__init__(num_classes, n_tokens_in, n_channels_in, n_tokens_out,
                                                    n_channels_out,
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
