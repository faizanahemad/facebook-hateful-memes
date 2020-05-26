import abc
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
from ...preprocessing import clean_text


class FasttextPooledModel(nn.Module):
    def __init__(self, classifer_dims, num_classes,
                 gaussian_noise=0.0, dropout=0.0, use_as_submodel=False,
                 **kwargs):
        super(FasttextPooledModel, self).__init__()
        fasttext_file = kwargs["fasttext_file"] if "fasttext_file" in kwargs else None
        fasttext_model = kwargs["fasttext_model"] if "fasttext_model" in kwargs else None
        assert fasttext_file is not None or fasttext_model is not None
        if fasttext_file is not None:
            self.text_model = fasttext.load_model(fasttext_file)
        else:
            self.text_model = fasttext_model

        projection = [nn.Linear(500, classifer_dims * 2), nn.LeakyReLU(),
                      nn.Linear(classifer_dims * 2, classifer_dims)]
        init_fc(projection[0], 'xavier_uniform_', "leaky_relu")
        init_fc(projection[2], 'xavier_uniform_', "linear")
        self.projection = nn.Sequential(*projection)
        layers = [GaussianNoise(gaussian_noise), nn.Linear(classifer_dims, classifer_dims, bias=True),
                  nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(classifer_dims, num_classes)]
        self.bpe = BPEmb(dim=100)
        self.cngram = CharNGram()
        init_fc(layers[1], 'xavier_uniform_', "leaky_relu")
        init_fc(layers[4], 'xavier_uniform_', "linear")
        self.classifier = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.use_as_submodel = use_as_submodel

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
        result = stack_and_pad_tensors([self.get_one_sentence_vector(tm, text) for text in texts], 64)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        res2 = stack_and_pad_tensors([self.get_one_sentence_vector(bpe, text) for text in texts], 64)
        res2 = res2 / res2.norm(dim=2, keepdim=True).clamp(min=1e-5)
        res3 = stack_and_pad_tensors([self.get_one_sentence_vector(cngram, text) for text in texts], 64)
        res3 = res3 / res3.norm(dim=2, keepdim=True).clamp(min=1e-5)
        result = torch.cat([result, res2, res3], 2)
        result = result / result.norm(dim=2, keepdim=True).clamp(min=1e-5)  # Normalize in word dimension
        return result

    def forward(self, texts: List[str], img, labels):
        projections, vectors = self.__get_scores__(texts, img)
        if self.use_as_submodel:
            loss = None
            preds = None
        else:
            logits = self.classifier(projections) if not self.use_as_submodel else None
            loss = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
            preds = logits.max(dim=1).indices
            logits = torch.softmax(logits, dim=1)
        return logits, preds, vectors, loss

    def __get_scores__(self, texts: List[str], img):
        projections = self.projection(self.get_sentence_vector(texts))
        return projections, projections.unsqueeze(1)




