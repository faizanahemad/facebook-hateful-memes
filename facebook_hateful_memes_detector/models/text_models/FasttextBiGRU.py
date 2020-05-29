import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
import fasttext
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors
from .FasttextPooled import FasttextPooledModel


class FasttextBiGRUModel(FasttextPooledModel):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0, use_as_submodel=False,
                 **kwargs):
        super(FasttextBiGRUModel, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout, use_as_submodel, **kwargs)
        gru_layers = kwargs["gru_layers"] if "gru_layers" in kwargs else 2
        gru_dropout = kwargs["gru_dropout"] if "gru_dropout" in kwargs else 0.1
        gru_dims = kwargs["gru_dims"] if "gru_dims" in kwargs else int(classifer_dims/2)
        lin1 = nn.Linear(gru_dims * 2, gru_dims * 4)
        init_fc(lin1, "leaky_relu")
        lin2 = nn.Linear(gru_dims * 4, classifer_dims)
        init_fc(lin2, "linear")
        self.projection = nn.Sequential(nn.Dropout(dropout), lin1, nn.LeakyReLU(), lin2)
        self.lstm = nn.Sequential(nn.GRU(500, gru_dims, gru_layers, batch_first=True, bidirectional=True, dropout=gru_dropout))
        # init_fc(self.lstm, 'linear')

    def __get_scores__(self, texts: List[str], img=None):
        vectors = self.get_word_vectors(texts)
        lstm_output, _ = self.lstm(vectors)
        lstm_output = self.projection(lstm_output)
        mean_projection = lstm_output.mean(1)
        return mean_projection, lstm_output

