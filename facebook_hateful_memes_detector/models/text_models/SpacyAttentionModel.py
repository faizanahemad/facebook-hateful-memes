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
import spacy

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_pos_tag_indices, pad_tensor
from .FasttextPooled import FasttextPooledModel


class SpacyAttentionModel(FasttextPooledModel):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0, use_as_submodel=False,
                 **kwargs):
        super(SpacyAttentionModel, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout, use_as_submodel, **kwargs)
        gru_layers = kwargs["gru_layers"] if "gru_layers" in kwargs else 2
        gru_dropout = kwargs["gru_dropout"] if "gru_dropout" in kwargs else 0.1
        gru_dims = kwargs["gru_dims"] if "gru_dims" in kwargs else int(classifer_dims/2)
        lin = nn.Linear(gru_dims * 2, classifer_dims)
        init_fc(lin, "xavier_uniform_", "linear")
        embedding_dim = 4
        self.embeds = nn.Embedding(20, embedding_dim)
        nn.init.normal_(self.embeds.weight, std=1 / embedding_dim)
        self.projection = lin
        self.lstm = nn.Sequential(nn.GRU(100, gru_dims, gru_layers, batch_first=True, bidirectional=True, dropout=gru_dropout))
        self.nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
        self.pdict = get_pos_tag_indices()

    def get_word_vectors(self, texts: List[str]):
        pdict = self.pdict
        nlp = self.nlp
        texts = list(nlp.pipe(texts, n_process=4))
        text_tensors = list(map(lambda x: torch.tensor(x.tensor), texts))
        text_tensors = stack_and_pad_tensors(text_tensors, 64)
        pos = stack_and_pad_tensors(list(map(lambda x: torch.tensor([pdict[token.pos_] for token in x]), texts)), 64)
        pos_emb = self.embeds(pos)
        result = torch.cat([text_tensors, pos_emb], 2)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        return result

    def __get_scores__(self, texts: List[str], img=None):
        vectors = self.get_word_vectors(texts)
        lstm_output, _ = self.lstm(vectors)
        lstm_output = self.projection(lstm_output)
        # lstm_output = lstm_output / lstm_output.norm(dim=2, keepdim=True).clamp(min=1e-5)
        mean_projection = lstm_output.mean(1)
        # mean_projection = mean_projection / mean_projection.norm(dim=1, keepdim=True).clamp(min=1e-5)
        return mean_projection, lstm_output
