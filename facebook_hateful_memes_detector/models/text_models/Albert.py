import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, BasicFeaturizer, TransformerFeaturizer
from .Fasttext1DCNN import Fasttext1DCNNModel
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel, DistilBertTokenizer, LongformerTokenizer
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification, DistilBertModel, LongformerModel
import torchvision.models as models
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb
from ...utils import get_device, GaussianNoise, random_word_mask, load_stored_params, ExpandContract, Transformer, PositionalEncoding, LambdaLayer, get_global, \
    get_regularization_layers, WordMasking
import os
import random
import math


class AlbertClassifer(Fasttext1DCNNModel):
    def __init__(self, classifier_dims, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False, **kwargs):
        embedding_dims = 768
        super(AlbertClassifer, self).__init__(classifier_dims, num_classes, embedding_dims, gaussian_noise, dropout,
                                              internal_dims, n_layers,
                                              featurizer, final_layer_builder,
                                              n_tokens_in, n_tokens_out, True, **kwargs)
        self.word_masking_proba = kwargs["word_masking_proba"] if "word_masking_proba" in kwargs else 0.0
        self.need_fasttext = "fasttext_vector_config" in kwargs
        if "fasttext_vector_config" in kwargs:
            import fasttext
            ftvc = kwargs["fasttext_vector_config"]
            gru_layers = ftvc.pop("gru_layers", 0)
            fasttext_crawl = fasttext.load_model("crawl-300d-2M-subword.bin")
            fasttext_wiki = fasttext.load_model("wiki-news-300d-1M-subword.bin")
            bpe = BPEmb(dim=200)
            cngram = CharNGram()
            self.word_vectorizers = dict(fasttext_crawl=fasttext_crawl, fasttext_wiki=fasttext_wiki, bpe=bpe, cngram=cngram)
            crawl_nn = ExpandContract(900, embedding_dims, dropout,
                                      use_layer_norm=True, unit_norm=False, groups=(4, 4))
            self.crawl_nn = crawl_nn
            n_tokens_in = n_tokens_in + (8 * int(self.n_tokens_in/(8*1.375) + 1))
            if gru_layers > 0:
                lstm = nn.Sequential(GaussianNoise(gaussian_noise),
                                     nn.GRU(embedding_dims, int(embedding_dims / 2), gru_layers, batch_first=True, bidirectional=True, dropout=dropout))
                pre_query_layer = nn.Sequential(lstm, LambdaLayer(lambda x: x[0]), nn.LayerNorm(embedding_dims))
            else:
                pre_query_layer = nn.LayerNorm(embedding_dims)
            self.pre_query_layer = pre_query_layer

        if not use_as_super:
            model = kwargs["model"] if "model" in kwargs else 'albert-base-v2'
            global_dir = get_global("models_dir")
            model = os.path.join(global_dir, model) if model in os.listdir(global_dir) else model
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModel.from_pretrained(model)
            print("Pick stored Model", model, "Model Class = ", type(self.model), "Tokenizer Class = ", type(self.tokenizer))
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
                self.attention_drop_proba = kwargs["attention_drop_proba"] if "attention_drop_proba" in kwargs else 0.0
                n_encoders = kwargs.pop("n_encoders", n_layers)
                n_decoders = kwargs.pop("n_decoders", n_layers)
                self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                        classifier_dims,
                                                        internal_dims, n_encoders, n_decoders,
                                                        gaussian_noise, dropout, self.attention_drop_proba)
            else:
                raise NotImplementedError()

            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)
        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.word_masking = WordMasking(tokenizer=self.tokenizer, word_masking_proba=self.word_masking_proba, **kwargs)
        self.reg_layers = get_regularization_layers(self)

    def fasttext_vectors(self, texts: List[str]):
        word_vectors = self.get_fasttext_vectors(texts, 8 * int(self.n_tokens_in/(8*1.375) + 1), **self.word_vectorizers)
        word_vectors = self.crawl_nn(word_vectors)
        word_vectors = self.pre_query_layer(word_vectors)
        return word_vectors

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        texts = self.word_masking(texts)
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def get_word_vectors(self, texts: List[str]):
        input_ids, attention_mask = self.tokenise(texts)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        if self.need_fasttext:
            fasttext_vectors = self.fasttext_vectors(texts)
            last_hidden_states = torch.cat((last_hidden_states, fasttext_vectors), 1)
        pooled_output = outputs[1]
        return last_hidden_states
