import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
import fasttext
from mmf.common import SampleList
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...training import calculate_auc_dice_loss
from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, ExpandContract, get_device, dict2sampleList
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer, BasicFeaturizer


class Fasttext1DCNNModel(nn.Module):
    def __init__(self, classifier_dims, num_classes, embedding_dims,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False,
                 **kwargs):
        super(Fasttext1DCNNModel, self).__init__()
        fasttext_file = kwargs[
            "fasttext_file"] if "fasttext_file" in kwargs else "crawl-300d-2M-subword.bin"  # "wiki-news-300d-1M-subword.bin"
        fasttext_model = kwargs["fasttext_model"] if "fasttext_model" in kwargs else None
        assert fasttext_file is not None or fasttext_model is not None or use_as_super
        self.num_classes = num_classes
        self.binary = num_classes == 2
        self.auc_loss = True
        self.dice = True
        self.n_tokens_in = n_tokens_in
        self.n_tokens_out = n_tokens_out

        if not use_as_super:
            if fasttext_file is not None:
                self.text_model = fasttext.load_model(fasttext_file)
            else:
                self.text_model = fasttext_model

            self.crawl_nn = ExpandContract(200 + 300 + 100, embedding_dims, dropout,
                                           use_layer_norm=True, unit_norm=False, groups=(8, 4))
            self.bpe = BPEmb(dim=200)
            self.cngram = CharNGram()

            if featurizer == "cnn":
                self.featurizer = CNN1DFeaturizer(n_tokens_in, embedding_dims, n_tokens_out, classifier_dims, internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "transformer":
                n_encoders = kwargs["n_encoders"] if "n_encoders" in kwargs else n_layers
                n_decoders = kwargs["n_decoders"] if "n_decoders" in kwargs else n_layers
                self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                        classifier_dims,
                                                        internal_dims, n_encoders, n_decoders, gaussian_noise, dropout)
            elif featurizer == "basic":
                self.featurizer = BasicFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifier_dims,
                                                  internal_dims, n_layers, gaussian_noise, dropout)

            elif featurizer == "gru":
                self.featurizer = GRUFeaturizer(n_tokens_in, embedding_dims, n_tokens_out, classifier_dims, internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()
            loss = kwargs["loss"] if "loss" in kwargs else None
            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, loss)

        self.reg_layers = [(c, c.p if hasattr(c, "p") else c.sigma) for c in self.children() if c.__class__ == GaussianNoise or c.__class__ == nn.Dropout]
        self.auc_loss_coef = kwargs["auc_loss_coef"] if "auc_loss_coef" in kwargs else 4.0
        self.dice_loss_coef = kwargs["dice_loss_coef"] if "dice_loss_coef" in kwargs else 2.0

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        texts = sampleList.text
        labels = torch.tensor(sampleList.label).to(get_device())
        # sample_weights = torch.tensor(sampleList.sample_weight, dtype=float).to(get_device())
        del sampleList
        vectors = self.get_word_vectors(texts)
        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels) if self.final_layer is not None else (None, None)

        if self.training:
            loss = calculate_auc_dice_loss(logits, labels, loss, self.auc_loss_coef, self.dice_loss_coef)
        return logits, vectors.mean(1), vectors, loss

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
        result = result.to(get_device())
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

    def get_word_vectors(self, texts: List[str]):

        # expected output # Bx64x512
        bpe = self.bpe
        cngram = self.cngram
        tm = self.text_model
        n_tokens_in = self.n_tokens_in
        result = stack_and_pad_tensors([self.get_one_sentence_vector(tm, text) for text in texts], n_tokens_in)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        res2 = stack_and_pad_tensors([self.get_one_sentence_vector(bpe, text) for text in texts], n_tokens_in)
        res2 = res2 / res2.norm(dim=2, keepdim=True).clamp(min=1e-5)
        res3 = stack_and_pad_tensors([self.get_one_sentence_vector(cngram, text) for text in texts], n_tokens_in)
        res3 = res3 / res3.norm(dim=2, keepdim=True).clamp(min=1e-5)
        result = torch.cat([result, res2, res3], 2)
        result = result.to(get_device())
        result = self.crawl_nn(result)
        return result

