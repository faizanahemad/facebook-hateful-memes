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
from .Fasttext1DCNN import Fasttext1DCNNModel
from ...utils import WordChannelReducer
from ..classifiers import CNN1DClassifier, GRUClassifier


class SpacyBiGRUModel(Fasttext1DCNNModel):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0,
                 embedding_dims=136,
                 internal_dims=512, n_layers=2,
                 classifier="gru",
                 use_as_super=False,
                 **kwargs):
        super(SpacyBiGRUModel, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout, True, **kwargs)
        gru_dims = kwargs["gru_dims"] if "gru_dims" in kwargs else int(classifer_dims/2)
        if not use_as_super:
            if classifier == "cnn":
                self.classifier = CNN1DClassifier(num_classes, 64, embedding_dims, 16, classifer_dims, internal_dims, None, gaussian_noise, dropout)
            elif classifier == "gru":
                self.classifier = GRUClassifier(num_classes, 64, embedding_dims, 16, classifer_dims, internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()
        # init_fc(self.lstm, 'linear')
        self.nlp = spacy.load("en_core_web_lg", disable=["ner"])
        self.pdict = get_all_tags()
        embedding_dim = 8
        self.tag_em = nn.Embedding(len(self.pdict)+1, embedding_dim)
        # init_fc(self.tag_em, "linear")
        nn.init.normal_(self.tag_em.weight, std=1 / embedding_dim)

        self.sw_em = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.sw_em.weight, std=1 / embedding_dim)
        self.projection = WordChannelReducer(gru_dims * 2, classifer_dims, 4)

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
