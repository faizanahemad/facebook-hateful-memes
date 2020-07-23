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

from ...training import calculate_auc_dice_loss, get_auc_dice_loss
from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, ExpandContract, get_device, dict2sampleList, load_stored_params
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
        fasttext_file = kwargs.pop("fasttext_file", "crawl-300d-2M-subword.bin")  # "wiki-news-300d-1M-subword.bin"
        fasttext_model = kwargs.pop("fasttext_model", None)
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
                self.attention_drop_proba = kwargs["attention_drop_proba"] if "attention_drop_proba" in kwargs else 0.0
                n_encoders = kwargs["n_encoders"] if "n_encoders" in kwargs else n_layers
                n_decoders = kwargs["n_decoders"] if "n_decoders" in kwargs else n_layers
                self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                        classifier_dims,
                                                        internal_dims, n_encoders, n_decoders,
                                                        gaussian_noise, dropout, self.attention_drop_proba)
            elif featurizer == "basic":
                self.featurizer = BasicFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifier_dims,
                                                  internal_dims, n_layers, gaussian_noise, dropout)

            elif featurizer == "gru":
                self.featurizer = GRUFeaturizer(n_tokens_in, embedding_dims, n_tokens_out, classifier_dims, internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()
            loss = kwargs["loss"] if "loss" in kwargs else None
            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)

        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.reg_layers = [(c, c.p if hasattr(c, "p") else c.sigma) for c in self.children() if c.__class__ == GaussianNoise or c.__class__ == nn.Dropout]
        self.auc_loss_coef = kwargs.pop("auc_loss_coef", 0.0)
        self.dice_loss_coef = kwargs.pop("dice_loss_coef", 0.0)
        self.auc_method = kwargs.pop("auc_method", 1)
        self.auc_dice_loss = get_auc_dice_loss(num_classes, self.dice_loss_coef, self.auc_loss_coef, auc_method=self.auc_method)

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
            loss += self.auc_dice_loss(logits, labels)
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

    @classmethod
    def get_one_sentence_vector(cls, tm, sentence):
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
        result = self.get_fasttext_vectors(texts, n_tokens_in, fasttext_crawl=tm, bpe=bpe, cngram=cngram,)
        result = self.crawl_nn(result)
        return result

    @classmethod
    def get_fasttext_vectors(cls, texts: List[str], n_tokens_in,
                             fasttext_crawl=None, fasttext_wiki=None,
                             bpe=None, cngram=None,):
        result = []
        if fasttext_crawl:
            res0 = stack_and_pad_tensors([cls.get_one_sentence_vector(fasttext_crawl, text) for text in texts], n_tokens_in)
            res0 = res0 / res0.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
            result.append(res0)
        if fasttext_wiki:
            res1 = stack_and_pad_tensors([cls.get_one_sentence_vector(fasttext_wiki, text) for text in texts], n_tokens_in)
            res1 = res1 / res1.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
            result.append(res1)
        if bpe:
            res2 = stack_and_pad_tensors([cls.get_one_sentence_vector(bpe, text) for text in texts], n_tokens_in)
            res2 = res2 / res2.norm(dim=2, keepdim=True).clamp(min=1e-5)
            result.append(res2)
        if cngram:
            res3 = stack_and_pad_tensors([cls.get_one_sentence_vector(cngram, text) for text in texts], n_tokens_in)
            res3 = res3 / res3.norm(dim=2, keepdim=True).clamp(min=1e-5)
            result.append(res3)
        result = torch.cat(result, 2)
        result = result.to(get_device())

        return result

