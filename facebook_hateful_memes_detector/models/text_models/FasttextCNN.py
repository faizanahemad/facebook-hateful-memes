import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from ...utils.sample import SampleList, Sample
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...training import get_auc_dice_loss
from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, ExpandContractV2, get_device, dict2sampleList, load_stored_params, \
    get_regularization_layers, WordMasking
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer, BasicFeaturizer
from ...training import fb_1d_loss_builder
import fasttext
import random


def random_whole_word_mask(text: str, probability: float) -> str:
    text = str(text)
    if probability == 0 or len(text) == 0 or len(text.split()) <= 2:
        return text
    tokens = text.split()
    new_tokens = []
    for idx, token in enumerate(tokens):
        prob = random.random()
        if prob < probability:
            prob /= probability
            if prob < 0.75 or len(token) <= 3:
                tks = '<mask>'
            else:
                tks = token
        else:
            tks = token
        new_tokens.append(tks)
    return " ".join(new_tokens)


class FasttextCNN(nn.Module):
    def __init__(self, classifier_dims, num_classes,
                 gaussian_noise, dropout, feature_dropout,
                 n_layers, device,
                 n_tokens_in=512, n_tokens_out=8,
                 **kwargs):
        super(FasttextCNN, self).__init__()
        fasttext_file = kwargs.pop("fasttext_file", "crawl-300d-2M-subword.bin")  # "wiki-news-300d-1M-subword.bin"
        fasttext_file_2 = kwargs.pop("fasttext_file_2", "wiki-news-300d-1M-subword.bin")  # "wiki-news-300d-1M-subword.bin"
        self.num_classes = num_classes
        self.binary = num_classes == 2
        self.n_tokens_in = n_tokens_in
        self.n_tokens_out = n_tokens_out
        self.device = device
        self.word_masking_proba = kwargs.pop("word_masking_proba", 0.0)
        embedding_dims = classifier_dims // 2
        internal_dims = classifier_dims

        self.text_model = fasttext.load_model(fasttext_file)
        self.text_model_2 = fasttext.load_model(fasttext_file_2)

        if fasttext_file_2 == "wiki-news-300d-1M-subword.bin":
            ft2_dim = 300
        else:
            ft2_dim = 512

        self.ft1_nn = ExpandContractV2(300, embedding_dims, dropout, feature_dropout)
        self.ft2_nn = ExpandContractV2(ft2_dim, embedding_dims, dropout, feature_dropout)
        self.bpe_nn = ExpandContractV2(300, embedding_dims, dropout, feature_dropout)
        self.cngram_nn = ExpandContractV2(100, embedding_dims, dropout, feature_dropout)
        self.input_nn = ExpandContractV2(4 * embedding_dims, internal_dims, dropout, feature_dropout)

        self.bpe = BPEmb(dim=300)
        self.cngram = CharNGram()

        self.featurizer = GRUFeaturizer(n_tokens_in, internal_dims, n_tokens_out, classifier_dims, internal_dims, n_layers, gaussian_noise, dropout)
        self.final_layer = fb_1d_loss_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)

        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])

    def forward(self, f1_vector, f2_vector, bpe_vector, cngram_vector,  labels=None):
        f1_vector = self.ft1_nn(f1_vector)
        f2_vector = self.ft2_nn(f2_vector)
        bpe_vector = self.bpe_nn(bpe_vector)
        cngram_vector = self.cngram_nn(cngram_vector)

        vectors = torch.cat((f1_vector, f2_vector, bpe_vector, cngram_vector), 2)
        vectors = self.input_nn(vectors)
        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels) if self.final_layer is not None else (None, None)
        logits = torch.softmax(logits, dim=1)
        predicted_labels = logits.max(dim=1).indices
        return logits, predicted_labels, labels, loss

    def get_sentence_vector(self, texts: List[str]):
        tm = self.text_model
        bpe = self.bpe
        cngram = self.cngram
        result = torch.tensor([tm.get_sentence_vector(text) for text in texts])
        result = result / result.norm(dim=1, keepdim=True).clamp(min=1e-5)  # Normalize in sentence dimension
        res2 = torch.stack([self.get_one_sentence_vector(bpe, text).mean(0) for text in texts])
        res2 = res2 / res2.norm(dim=1, keepdim=True).clamp(min=1e-5)
        res3 = torch.stack([cngram[text] for text in texts])
        res3 = res3 / res3.norm(dim=1, keepdim=True).clamp(min=1e-5)
        result = torch.cat([result, res2, res3], 1)
        result = result / result.norm(dim=1, keepdim=True).clamp(min=1e-5)
        return result

    def get_one_sentence_vector(self, tm, sentence):

        tokens = fasttext.tokenize(sentence)
        if isinstance(tm, fasttext.FastText._FastText):
            result = torch.tensor([tm[t] for t in tokens])
        elif isinstance(tm, torchnlp.word_to_vector.char_n_gram.CharNGram):
            result = torch.stack([tm[t] for t in tokens])
        else:
            result = tm[tokens]
        return result

    def get_fasttext_vectors(self, texts: List[str]):
        if self.training:
            texts = [random_whole_word_mask(t, self.word_masking_proba) for t in texts]
        n_tokens_in = self.n_tokens_in
        result = []

        res0 = stack_and_pad_tensors([self.get_one_sentence_vector(self.text_model, text) for text in texts], n_tokens_in)
        res0 = res0 / res0.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        result.append(res0)

        res1 = stack_and_pad_tensors([self.get_one_sentence_vector(self.text_model_2, text) for text in texts], n_tokens_in)
        res1 = res1 / res1.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        result.append(res1)

        res2 = stack_and_pad_tensors([self.get_one_sentence_vector(self.bpe, text) for text in texts], n_tokens_in)
        res2 = res2 / res2.norm(dim=2, keepdim=True).clamp(min=1e-5)
        result.append(res2)

        res3 = stack_and_pad_tensors([self.get_one_sentence_vector(self.cngram, text) for text in texts], n_tokens_in)
        res3 = res3 / res3.norm(dim=2, keepdim=True).clamp(min=1e-5)
        result.append(res3)
        result = [r.to(self.device) for r in result]

        return result

