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

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_pos_tag_indices, pad_tensor, \
    get_penn_treebank_pos_tag_indices, get_all_tags
from ...utils import get_universal_deps_indices
from .FasttextPooled import FasttextPooledModel
from ..ibm_max import ModelWrapper


class GensimLangFeaturesModel1DCNN(FasttextPooledModel):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0, use_as_submodel=False, use_as_super=False,
                 **kwargs):
        super(GensimLangFeaturesModel1DCNN, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout, use_as_submodel, True, **kwargs)

        if not use_as_super:
            pass

    def get_word_and_text_lengths(self):
        pass

    def get_word_vectors(self, texts: List[str]):
        result = super().get_word_vectors(texts)

    def __get_scores__(self, texts: List[str], img=None):
        vectors = self.get_word_vectors(texts)

