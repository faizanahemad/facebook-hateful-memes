import abc
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
    def __init__(self, gru_layers, gru_dropout, hidden_dims, num_classes, fasttext_file=None, fasttext_model=None, gaussian_noise=0.0):
        super(FasttextBiGRUModel, self).__init__(hidden_dims, num_classes, fasttext_file, fasttext_model, gaussian_noise)
        self.lstm = nn.GRU(500, hidden_dims, gru_layers, batch_first=True, bidirectional=True, dropout=gru_dropout)
        layers = [GaussianNoise(gaussian_noise), nn.Linear(hidden_dims * 2, hidden_dims, bias=False),
                  nn.LeakyReLU(), nn.Linear(hidden_dims, num_classes)]
        init_fc(layers[1], 'xavier_uniform_', "leaky_relu")
        init_fc(layers[3], 'xavier_uniform_', "linear")
        self.classifier = nn.Sequential(*layers)

    def predict_proba(self, texts: List[str], img=None):
        vectors = self.get_word_vectors(texts)
        lstm_output, _ = self.lstm(vectors)
        lstm_output = lstm_output.mean(1)
        logits = self.classifier(lstm_output)
        return logits

    @staticmethod
    def build(**kwargs):
        gru_layers = kwargs["gru_layers"]
        gru_dropout = kwargs["gru_dropout"]
        hidden_dims = kwargs["hidden_dims"]
        num_classes = kwargs["num_classes"]
        fasttext_file = kwargs["fasttext_file"] if "fasttext_file" in kwargs else None
        fasttext_model = kwargs["fasttext_model"] if "fasttext_model" in kwargs else None
        gaussian_noise = kwargs["gaussian_noise"] if "gaussian_noise" in kwargs else 0.0
        return FasttextBiGRUModel(gru_layers, gru_dropout, hidden_dims, num_classes, fasttext_file, fasttext_model, gaussian_noise)

