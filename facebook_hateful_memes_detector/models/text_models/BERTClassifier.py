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
from ...training import fb_1d_loss_builder
import os
import random
import math


class BERTClassifier(nn.Module):
    def __init__(self, classifier_dims, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 device,
                 n_tokens_in=64, n_tokens_out=16,
                 **kwargs):
        embedding_dims = 768
        super(BERTClassifier, self).__init__()
        self.word_masking_proba = kwargs["word_masking_proba"] if "word_masking_proba" in kwargs else 0.0
        self.mlm_probability = self.word_masking_proba
        self.n_tokens_in = n_tokens_in
        self.token_cache = kwargs.pop("token_cache", None)
        self.force_masking = False

        model = kwargs["model"] if "model" in kwargs else 'albert-base-v2'
        global_dir = get_global("models_dir")
        model = os.path.join(global_dir, model) if model in os.listdir(global_dir) else model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        print("Pick stored Model", model, "Model Class = ", type(self.model), "Tokenizer Class = ", type(self.tokenizer))

        self.attention_drop_proba = kwargs["attention_drop_proba"] if "attention_drop_proba" in kwargs else 0.0
        n_encoders = kwargs.pop("n_encoders", n_layers)
        n_decoders = kwargs.pop("n_decoders", n_layers)
        self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                classifier_dims,
                                                internal_dims, n_encoders, n_decoders,
                                                gaussian_noise, dropout, self.attention_drop_proba)

        self.final_layer = fb_1d_loss_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)
        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.word_masking = WordMasking(tokenizer=self.tokenizer, **kwargs)
        self.device=device

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def tokenise(self, ids, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        if self.token_cache is None:
            converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
            input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        else:
            input_ids, attention_mask = zip(*[self.token_cache[id] for id in ids])

        input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)
        if self.training or self.force_masking:
            input_ids, _ = self.mask_tokens(input_ids)
        return input_ids.to(self.device), attention_mask.to(self.device)


    def get_word_vectors(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, input_ids, attention_mask, labels=None):
        vectors = self.get_word_vectors(input_ids, attention_mask)
        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels) if self.final_layer is not None else (None, None)
        logits = torch.softmax(logits, dim=1)
        predicted_labels = logits.max(dim=1).indices
        return logits, predicted_labels, labels, loss
