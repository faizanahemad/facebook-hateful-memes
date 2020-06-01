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
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BytePairEmbeddings, CharacterEmbeddings, WordEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
import json
import csv
import numpy as np

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import stanza


import spacy

from .LangFeatures import LangFeaturesModel
from .Fasttext1DCNN import Fasttext1DCNNModel
from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_pos_tag_indices, pad_tensor, \
    get_penn_treebank_pos_tag_indices, get_all_tags
from ...utils import get_universal_deps_indices
from .FasttextPooled import FasttextPooledModel
import gensim.downloader as api
from ...utils import WordChannelReducer, Transpose, Squeeze
from ..classifiers import CNN1DClassifier, GRUClassifier


class GensimLangFeatures1DCNNModel(Fasttext1DCNNModel):
    def __init__(self, classifer_dims, num_classes, embedding_dims,
                 gaussian_noise=0.0, dropout=0.0,
                 internal_dims=512, n_layers=2,
                 classifier="cnn",
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False,
                 **kwargs):
        super(GensimLangFeatures1DCNNModel, self).__init__(classifer_dims, num_classes, embedding_dims, gaussian_noise, dropout,
                                                           internal_dims, n_layers,
                                                           classifier,
                                                           n_tokens_in, n_tokens_out,
                                                           True, **kwargs)
        models = [api.load("glove-twitter-50"), api.load("glove-wiki-gigaword-50"),
                  api.load("word2vec-google-news-300"), api.load("conceptnet-numberbatch-17-06-300")]
        self.models = dict(zip(range(len(models)), models))
        self.spacy = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner"])
        embedding_dims = 700
        if not use_as_super:
            if classifier == "cnn":
                self.classifier = CNN1DClassifier(num_classes, n_tokens_in, embedding_dims, n_tokens_out, classifer_dims, internal_dims, None, gaussian_noise, dropout)
            elif classifier == "gru":
                self.classifier = GRUClassifier(num_classes, n_tokens_in, embedding_dims, n_tokens_out, classifer_dims, internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()

        # init_fc(self.lstm, 'linear')

    def get_one_sentence_vector(self, i, m, sentence):
        result = [m[t] if t in m else np.zeros(m.vector_size) for t in sentence]
        return torch.tensor(result, dtype=float)

    def get_word_vectors(self, texts: List[str]):
        n_tokens_in = self.n_tokens_in
        # wv1 = super().get_word_vectors(texts)
        texts = list(self.spacy.pipe(texts, n_process=4))
        texts = list(map(lambda x: list(map(str, x)), texts))
        result = []
        for i, m in self.models.items():
            r = stack_and_pad_tensors([self.get_one_sentence_vector(i, m, text) for text in texts], n_tokens_in)
            result.append(r)
        result = [r.float() for r in result]
        result = torch.cat(result, 2)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        return result






