import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, BasicFeaturizer, TransformerFeaturizer
from .Fasttext1DCNN import Fasttext1DCNNModel
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
import torchvision.models as models
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb
from ...utils import get_device, GaussianNoise, random_word_mask, load_stored_params, ExpandContract, Transformer, PositionalEncoding, LambdaLayer
import os
import random
import math


class AlbertClassifer(Fasttext1DCNNModel):
    def __init__(self, classifier_dims, num_classes, embedding_dims,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False, **kwargs):
        super(AlbertClassifer, self).__init__(classifier_dims, num_classes, embedding_dims, gaussian_noise, dropout,
                                              internal_dims, n_layers,
                                              featurizer, final_layer_builder,
                                              n_tokens_in, n_tokens_out, True, **kwargs)
        assert n_tokens_in % n_tokens_out == 0
        model = kwargs["model"] if "model" in kwargs else 'albert-base-v2'
        self.word_masking_proba = kwargs["word_masking_proba"] if "word_masking_proba" in kwargs else 0.0
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.need_fasttext = "fasttext_vector_config" in kwargs
        if "fasttext_vector_config" in kwargs:
            import fasttext
            ftvc = kwargs["fasttext_vector_config"]
            gru_layers = ftvc.pop("gru_layers", 0)
            n_decoders = ftvc.pop("n_decoders", 2)
            fasttext_crawl = fasttext.load_model("crawl-300d-2M-subword.bin")
            fasttext_wiki = fasttext.load_model("wiki-news-300d-1M-subword.bin")
            bpe = BPEmb(dim=200)
            cngram = CharNGram()
            self.word_vectorizers = dict(fasttext_crawl=fasttext_crawl, fasttext_wiki=fasttext_wiki, bpe=bpe, cngram=cngram)
            crawl_nn = ExpandContract(900, embedding_dims, dropout,
                                      use_layer_norm=True, unit_norm=False, groups=(4, 4))

            if gru_layers > 0:
                lstm = nn.Sequential(GaussianNoise(gaussian_noise),
                                     nn.GRU(embedding_dims, int(embedding_dims / 2), gru_layers, batch_first=True, bidirectional=True, dropout=dropout))
                pre_query_layer = nn.Sequential(crawl_nn, lstm, nn.LayerNorm(embedding_dims), LambdaLayer(lambda x: x.transpose(0, 1)))
            else:
                pos_encoder = PositionalEncoding(embedding_dims, dropout)
                pre_query_layer = nn.Sequential(crawl_nn, LambdaLayer(lambda x: x.transpose(0, 1) * math.sqrt(embedding_dims)),
                                                pos_encoder, nn.LayerNorm(embedding_dims))
            self.pre_query_layer = pre_query_layer
            self.query_layer = Transformer(embedding_dims, 8, 0, n_decoders, embedding_dims*4, dropout, gaussian_noise)
            # We need gru+decoder support
            # n_internal_dims modification
            # support for selecting which fasttext entities we want
            # layer norm them and append extra dims
            pass
        if not use_as_super:
            if featurizer == "cnn":
                self.featurizer = CNN1DFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifier_dims, internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "gru":
                self.featurizer = GRUFeaturizer(n_tokens_in, embedding_dims, n_tokens_out, classifier_dims,
                                                internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "basic":
                self.featurizer = BasicFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                  classifier_dims,
                                                  internal_dims, n_layers, gaussian_noise, dropout)
            elif featurizer == "transformer":
                n_encoders = kwargs.pop("n_encoders", n_layers)
                n_decoders = kwargs.pop("n_decoders", n_layers)
                self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                        classifier_dims,
                                                        internal_dims, n_encoders, n_decoders, gaussian_noise, dropout)
            else:
                raise NotImplementedError()

            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)
        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.reg_layers = [(c, c.p if hasattr(c, "p") else c.sigma) for c in self.children() if c.__class__ == GaussianNoise or c.__class__ == nn.Dropout]

    def fasttext_vectors(self, texts: List[str], last_hidden_states: torch.Tensor):
        word_vectors = self.get_fasttext_vectors(texts, 8 * int(self.n_tokens_in/(8*1.375) + 1), **self.word_vectorizers)
        word_vectors = self.pre_query_layer(word_vectors)
        last_hidden_states = last_hidden_states.transpose(0, 1)
        word_vectors, _ = self.query_layer(word_vectors, last_hidden_states)
        return word_vectors

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        if self.training and self.word_masking_proba > 0:
            texts = [random_word_mask(t, tokenizer, self.word_masking_proba) for t in texts]
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def get_word_vectors(self, texts: List[str]):
        input_ids, attention_mask = self.tokenise(texts)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        if self.need_fasttext:
            fasttext_vectors = self.fasttext_vectors(texts, last_hidden_states)
            last_hidden_states = (last_hidden_states + fasttext_vectors) / 2
        pooled_output = outputs[1]
        return last_hidden_states
