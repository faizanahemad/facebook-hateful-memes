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

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, ExpandContract
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
        self.auc_loss = False
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
                self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                        classifier_dims,
                                                        internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "basic":
                self.featurizer = BasicFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifier_dims,
                                                  internal_dims, n_layers, gaussian_noise, dropout)

            elif featurizer == "gru":
                self.featurizer = GRUFeaturizer(n_tokens_in, embedding_dims, n_tokens_out, classifier_dims, internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()
            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, )

    def forward(self, sampleList: SampleList):
        texts = sampleList.text
        img = sampleList.image
        labels = sampleList.label
        sample_weights = sampleList.sample_weight

        vectors = self.get_word_vectors(texts)
        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels) if self.final_layer is not None else (None, None)
        if self.binary and self.training and self.auc_loss:
                # aucroc loss
                probas = logits[:, 1]
                pos_probas = labels * probas
                neg_probas = (1-labels) * probas
                pos_probas = pos_probas[pos_probas > self.eps]
                pos_probas_min = pos_probas.min()
                neg_probas_max = neg_probas.max() # torch.topk(neg_probas, 5, 0).values
                loss_1 = F.relu(neg_probas - pos_probas_min).mean()
                loss_2 = F.relu(neg_probas_max - pos_probas).mean()
                auc_loss = (loss_1 + loss_2)/2

                loss = 1.0 * loss + 0.0 * auc_loss

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
        result = self.crawl_nn(result)
        return result

