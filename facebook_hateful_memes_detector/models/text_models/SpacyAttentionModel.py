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

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_pos_tag_indices, pad_tensor, \
    get_penn_treebank_pos_tag_indices, get_all_tags
from ...utils import get_universal_deps_indices
from .FasttextPooled import FasttextPooledModel


class SpacyAttentionModel(FasttextPooledModel):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0, use_as_submodel=False,
                 **kwargs):
        super(SpacyAttentionModel, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout, use_as_submodel, True, **kwargs)
        gru_layers = kwargs["gru_layers"] if "gru_layers" in kwargs else 2
        gru_dropout = kwargs["gru_dropout"] if "gru_dropout" in kwargs else 0.1
        gru_dims = kwargs["gru_dims"] if "gru_dims" in kwargs else int(classifer_dims/2)
        lin1 = nn.Linear(gru_dims * 2, gru_dims * 4)
        init_fc(lin1, "leaky_relu")
        lin2 = nn.Linear(gru_dims * 4, classifer_dims)
        init_fc(lin2, "linear")

        self.projection = nn.Sequential(nn.Dropout(dropout), lin1, nn.LeakyReLU(), lin2)
        self.lstm = nn.Sequential(nn.GRU(136, gru_dims, gru_layers, batch_first=True, bidirectional=True, dropout=gru_dropout))
        # init_fc(self.lstm, 'linear')
        self.nlp = spacy.load("en_core_web_lg", disable=["ner"])
        self.pdict = get_all_tags()
        embedding_dim = 8
        self.tag_em = nn.Embedding(len(self.pdict)+1, embedding_dim)
        # init_fc(self.tag_em, "linear")
        nn.init.normal_(self.tag_em.weight, std=1 / embedding_dim)

        self.sw_em = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.sw_em.weight, std=1 / embedding_dim)

    def get_word_vectors(self, texts: List[str]):
        pdict = self.pdict
        nlp = self.nlp
        texts = list(nlp.pipe(texts, n_process=4))
        text_tensors = list(map(lambda x: torch.tensor(x.tensor), texts))
        text_tensors = stack_and_pad_tensors(text_tensors, 64)
        pos = stack_and_pad_tensors(list(map(lambda x: torch.tensor([pdict[token.pos_.lower()] for token in x]), texts)), 64)
        pos_emb = self.tag_em(pos)
        #
        tag = stack_and_pad_tensors(list(map(lambda x: torch.tensor([pdict[token.tag_.lower()] for token in x]), texts)), 64)
        tag_emb = self.tag_em(tag)

        dep = stack_and_pad_tensors(list(map(lambda x: torch.tensor([pdict[token.dep_.lower()] for token in x]), texts)), 64)
        dep_emb = self.tag_em(dep)

        sw = stack_and_pad_tensors(list(map(lambda x: torch.tensor([int(token.is_stop) for token in x]), texts)), 64)
        sw_emb = self.sw_em(sw)

        ner = stack_and_pad_tensors(
            list(map(lambda x: torch.tensor([pdict[token.ent_type_.lower()] for token in x]), texts)), 64)
        ner_emb = self.tag_em(ner)

        result = torch.cat([text_tensors, pos_emb, tag_emb, dep_emb, sw_emb, ner_emb], 2)
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
