import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..classifiers import CNN1DClassifier, GRUClassifier
from .Fasttext1DCNN import Fasttext1DCNNModel
import torchvision.models as models


class Albert(nn.Module):
    def __init__(self, max_length=64, output_length=16):
        from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
        super(Albert, self).__init__()
        assert max_length % output_length == 0
        assert output_length % 2 == 0
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.model = AlbertModel.from_pretrained('albert-base-v2')
        self.conv1d = nn.Conv1d(max_length, output_length, 1, stride=1, padding=0, dilation=1, groups=int(output_length/2), bias=False, padding_mode='zeros')
        self.max_length = max_length
        self.output_length = output_length

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        max_length = self.max_length
        m = lambda x: tokenizer.encode_plus(x, add_special_tokens=True, pad_to_max_length=True, max_length=128)
        input_ids, attention_mask = zip(*[(d['input_ids'], d['attention_mask']) for d in map(m, texts)])
        return torch.tensor(input_ids), torch.tensor(attention_mask)

    def compose(self, texts: List[str], *args, **kwargs):
        input_ids, attention_mask = self.tokenise(texts)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        pooled_output = outputs[1]
        last_hidden_states = self.conv1d(last_hidden_states)

    def forward(self, texts: List[str], img):
        pass

    def predict(self, texts: List[str], img):
        pass

    def predict_proba(self, texts: List[str], img):
        pass

    def __call__(self, *args, **kwargs):
        pass


