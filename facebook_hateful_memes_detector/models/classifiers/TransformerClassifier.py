from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from .BaseClassifier import BaseClassifier


class TransformerClassifier(BaseClassifier):
    def __init__(self, num_classes, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims,
                 gaussian_noise=0.0, dropout=0.0):
        super(TransformerClassifier, self).__init__(num_classes, n_tokens_in, n_channels_in, n_tokens_out,
                                                    n_channels_out,
                                                    n_internal_dims, gaussian_noise, dropout)
