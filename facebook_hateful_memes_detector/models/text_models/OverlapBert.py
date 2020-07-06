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
from ...utils import get_device, clean_memory


class OverlapbertModel(Fasttext1DCNNModel):
    def __init__(self, classifier_dims, num_classes, embedding_dims,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=768, n_overlap=128, n_tokens_out=16,
                 use_as_super=False, **kwargs):
        super(OverlapbertModel, self).__init__(classifier_dims, num_classes, embedding_dims, gaussian_noise, dropout,
                                                   internal_dims, n_layers,
                                                   featurizer, final_layer_builder,
                                                   n_tokens_in, n_tokens_out, True, **kwargs)
        assert n_tokens_in % n_tokens_out == 0
        assert n_tokens_in % n_overlap == 0
        assert n_overlap % 4 == 0
        model = kwargs["model"] if "model" in kwargs else 'albert-base-v2'
        self.n_overlap = n_overlap
        finetune = kwargs["finetune"] if "finetune" in kwargs else False
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.first_model_tokens = min(self.tokenizer.model_max_length, n_tokens_in)
        self.second_model_tokens = n_tokens_in - self.tokenizer.model_max_length + n_overlap + int(n_overlap/2)
        assert self.second_model_tokens <= self.tokenizer.model_max_length
        self.model = AutoModel.from_pretrained(model)
        self.finetune = finetune
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False
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
                n_encoders = kwargs["n_encoders"] if "n_encoders" in kwargs else n_layers
                n_decoders = kwargs["n_decoders"] if "n_decoders" in kwargs else n_layers
                self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                        classifier_dims,
                                                        internal_dims, n_encoders, n_decoders, gaussian_noise, dropout)
            else:
                raise NotImplementedError()

            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, )

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        first_model_tokens = self.first_model_tokens
        n_overlap = self.n_overlap
        model_max_length = self.tokenizer.model_max_length
        ovl1 = model_max_length - n_overlap
        ovl2 = int(n_overlap/2)
        m = lambda x: tokenizer.encode_plus(x, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = zip(*[(d['input_ids'], d['attention_mask']) for d in map(m, texts)])
        input_ids, attention_mask = torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())
        input_ids_extend = torch.cat((input_ids[:, :ovl2], input_ids[:, ovl1:]), 1)
        attention_mask_extend = torch.cat((attention_mask[:, :ovl2], attention_mask[:, ovl1:]), 1)
        return input_ids[:, :first_model_tokens], attention_mask[:, :first_model_tokens], input_ids_extend, attention_mask_extend

    def get_word_vectors(self, texts: List[str]):
        input_ids, attention_mask, input_ids_extend, attention_mask_extend = self.tokenise(texts)
        del texts
        if self.finetune:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            del input_ids
            del attention_mask
            clean_memory()
            outputs_extend = self.model(input_ids_extend, attention_mask=attention_mask_extend)
        else:
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                del input_ids
                del attention_mask
                clean_memory()
                outputs_extend = self.model(input_ids_extend, attention_mask=attention_mask_extend)
        clean_memory()
        last_hidden_states = torch.cat((outputs[0], outputs_extend[0]), dim=1)
        return last_hidden_states
