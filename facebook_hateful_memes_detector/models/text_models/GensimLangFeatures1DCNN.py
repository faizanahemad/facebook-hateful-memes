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


class GensimLangFeatures1DCNNModel(LangFeaturesModel):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0, use_as_submodel=False, embedding_dims=500, cnn_dims=512, use_as_super=False,
                 **kwargs):
        super(GensimLangFeatures1DCNNModel, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout, use_as_submodel, True, **kwargs)
        models = [api.load("glove-twitter-50"), api.load("glove-wiki-gigaword-50"),
                  api.load("word2vec-google-news-300"), api.load("conceptnet-numberbatch-17-06-300")]
        self.models = dict(zip(range(len(models)), models))
        self.spacy = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner"])
        conv1 = nn.Conv1d(embedding_dims, cnn_dims, 3, 1, padding=1, groups=4, bias=False)
        init_fc(conv1, "leaky_relu")
        mp = nn.MaxPool1d(2)
        conv2 = nn.Conv1d(cnn_dims, cnn_dims * 2, 3, 1, padding=1, groups=4, bias=False)
        init_fc(conv2, "leaky_relu")
        conv3 = nn.Conv1d(cnn_dims * 2, cnn_dims, 3, 1, padding=1, groups=4, bias=False)
        init_fc(conv3, "leaky_relu")
        relu = nn.LeakyReLU()
        dropout = nn.Dropout(dropout)
        gn = GaussianNoise(gaussian_noise)
        self.conv = nn.Sequential(gn, Transpose(), conv1, relu, dropout, mp,
                                  conv2, relu, dropout, mp,
                                  conv3, relu, dropout, mp,
                                  Transpose())
        self.classifier = nn.Sequential(Transpose(), nn.Conv1d(cnn_dims, num_classes, 8, 1, padding=0, groups=1, bias=False),
                                        Squeeze())
        # init_fc(self.lstm, 'linear')

    def __get_scores__(self, texts: List[str], img=None):
        vectors = self.get_word_vectors(texts)
        conv_out = self.conv(vectors)
        mean_projection = conv_out.mean(1)
        return mean_projection, conv_out

    def forward(self, texts: List[str], img, labels):
        projections, vectors = self.__get_scores__(texts, img)
        if self.use_as_submodel:
            loss = None
            preds = None
            logits = None
        else:
            logits = self.classifier(vectors) if not self.use_as_submodel else None
            loss = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
            preds = logits.max(dim=1).indices
            logits = torch.softmax(logits, dim=1)

        return logits, preds, projections, vectors, loss

    def get_one_sentence_vector(self, i, m, sentence):
        result = [m[t] if t in m else np.zeros(m.vector_size) for t in sentence]
        return torch.tensor(result, dtype=float)

    def get_word_vectors(self, texts: List[str]):
        wv1 = super().get_word_vectors(texts)
        texts = list(self.spacy.pipe(texts, n_process=4))
        texts = list(map(lambda x: list(map(str, x)), texts))
        result = [wv1]
        for i, m in self.models.items():
            r = stack_and_pad_tensors([self.get_one_sentence_vector(i, m, text) for text in texts], 64)
            result.append(r)
        result = [r.float() for r in result]
        result = torch.cat(result, 2)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        return result






