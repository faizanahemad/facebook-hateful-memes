import torchvision.models as models

# keep image as 2d or unroll
# Take a text and image model already pretrained, join them here with their get_vectors method
# Provide option to train e2e without pretrained as well
# Provide option for finetune vs full train

import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
import fasttext
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors
from ..text_models import FasttextPooledModel, Fasttext1DCNNModel
from ..classifiers import CNN1DClassifier, GRUClassifier, CNN1DSimple, TransformerClassifier, TransformerEnsembleClassifier


class TextImageAttentionModel(FasttextPooledModel):
    def __init__(self, classifer_dims, num_classes, embedding_dims,
                 gaussian_noise=0.0, dropout=0.0,
                 internal_dims=512, n_layers=2,
                 text_model="fasttext", # or "lang" or "albert" or actualmodel
                 classifier="transformer",
                 vision_model="resnet18",
                 n_tokens_in=64, n_tokens_out=16,
                 use_as_super=False,
                 **kwargs):
        super(TextImageAttentionModel, self).__init__(classifer_dims, num_classes, gaussian_noise, dropout,
                                                 n_tokens_in, n_tokens_out, True, **kwargs)

        self.classifier_type = classifier
        self.vision_model_type = vision_model
        if type(text_model) == str:
            if text_model == "fasttext":
                self.text_model = Fasttext1DCNNModel(classifer_dims, num_classes, gaussian_noise,
                                                     dropout, n_tokens_in, n_tokens_out, True, **kwargs)
            elif text_model == "lang":
                pass
            elif text_model == "albert":
                pass
            else:
                raise NotImplementedError()
        else:
            self.text_model = text_model

        if not use_as_super:
            if classifier == "transformer":
                self.classifier = TransformerClassifier(num_classes, n_tokens_in, embedding_dims, n_tokens_out,
                                                        classifer_dims,
                                                        internal_dims, n_layers, gaussian_noise, dropout)
            elif classifier == "transformer_ensemble":
                self.classifier = TransformerEnsembleClassifier(dict(text=dict(), image=dict()),
                                                                num_classes, n_tokens_out, classifer_dims,
                                                                internal_dims, n_layers, gaussian_noise, dropout)
            else:
                raise NotImplementedError()

    def forward(self, texts: List[str], img, labels, sample_weights=None):
        vectors = self.get_word_vectors(texts)
        logits, vectors = self.classifier(vectors)

        loss = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
        preds = logits.max(dim=1).indices
        logits = torch.softmax(logits, dim=1)

        return logits, preds, vectors.mean(1), vectors, loss


