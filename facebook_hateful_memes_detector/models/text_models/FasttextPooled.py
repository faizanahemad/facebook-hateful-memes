import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import fasttext
from ..utils import init_fc, clean_text


class FasttextPooledModel(nn.Module):
    def __init__(self, fasttext_file, in_dims, hidden_dims, num_classes):
        super(FasttextPooledModel, self).__init__()
        self.text_model = fasttext.load_model(fasttext_file)
        layers = [nn.Linear(in_dims, hidden_dims, bias=True), nn.LeakyReLU(), nn.Linear(hidden_dims, num_classes)]
        init_fc(layers[0], "leaky_relu")
        init_fc(layers[2], "linear")
        self.classifier = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()

    def get_sentence_vector(self, texts: List[str]):
        result = [list(self.text_model.get_sentence_vector(clean_text(text))) for text in texts]
        for i, r in enumerate(result):
            if np.sum(r[0:5]) == 0:
                result[i] = list(np.random.randn(self.in_dims))
        return torch.tensor(result)

    def forward(self, texts: List[str], img=None, labels=None):
        logits = self.predict_proba(texts, img)
        if labels is not None:
            labels = torch.tensor(labels)
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss
        else:
            return logits

    def predict(self, texts: List[str], img=None):
        logits = self.predict_proba(texts, img)
        return logits.max(dim=1).indices

    def predict_proba(self, texts: List[str], img=None):
        vectors = self.get_sentence_vector(texts)
        logits = self.classifier(vectors)
        return logits



