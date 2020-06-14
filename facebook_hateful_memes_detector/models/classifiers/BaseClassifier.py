from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F


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
