import torchvision
import torchvision.models as models

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
from ..classifiers import CNN1DClassifier, GRUClassifier, CNN1DSimple, TransformerClassifier, TransformerEnsembleClassifier


class ImageFullTextConvMidFusionModel(nn.Module):
    def __init__(self, image_model, text_model,
                 text_in_channels, num_classes,
                 finetune_image_model=False, finetune_text_model=False,
                 gaussian_noise=0.0, dropout=0.0,
                 internal_dims=128,
                 **kwargs):
        super(ImageFullTextConvMidFusionModel, self).__init__()
        if type(image_model) == str:
            if image_model == "torchvision_resnet18":
                im_model = torchvision.models.resnet18(pretrained=True)
                resnet_layers = list(im_model.children())[:-2]
                l1 = nn.Conv2d(512, internal_dims * 2, 1, 1, padding=0, groups=0, bias=False)
                init_fc(l1, "leaky_relu")
                l2 = nn.Conv2d(internal_dims * 2, internal_dims, 3, 1, padding=1, groups=4, bias=False)
                init_fc(l2, "leaky_relu")
                resnet_layers = resnet_layers + [nn.Dropout(dropout), l1, nn.LeakyReLU(),
                                                 GaussianNoise(gaussian_noise), l2, nn.LeakyReLU()]
                resnet18 = nn.Sequential(*resnet_layers)
                self.im_model = resnet18
                self.imf_width = 7

            else:
                raise NotImplementedError()
        if not finetune_image_model:
            for param in self.im_model.parameters():
                param.requires_grad = False

        if not finetune_text_model:
            for param in text_model.parameters():
                param.requires_grad = False
        self.text_model = text_model
        l1 = nn.Conv2d(text_in_channels, internal_dims, 1, 1, padding=0, groups=0, bias=False)
        init_fc(l1, "leaky_relu")
        self.text_channel_reducer = nn.Sequential(nn.Dropout(dropout), l1, nn.LeakyReLU(), nn.LayerNorm(internal_dims))

        l2 = nn.Conv2d(internal_dims * 2, internal_dims * 4, 3, 1, padding=1, groups=0, bias=False)
        init_fc(l2, "leaky_relu")
        l3 = nn.Conv2d(internal_dims * 4, internal_dims, 3, 1, padding=0, groups=0, bias=False)
        init_fc(l3, "linear")
        self.featurizer = nn.Sequential(nn.Dropout(dropout), l2, nn.LeakyReLU(), GaussianNoise(gaussian_noise), l3) # 5x5 for resnet18 from 7x7

        l2 = nn.Conv2d(internal_dims, internal_dims, 3, 1, padding=0, groups=0, bias=False)
        init_fc(l2, "leaky_relu")
        l3 = nn.Conv2d(internal_dims, num_classes, 3, 1, padding=0, groups=0, bias=False)
        init_fc(l3, "linear")
        self.classifier = nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout), l2, nn.LeakyReLU(), l3, nn.AdaptiveAvgPool2d(1))

    def forward(self, texts: List[str], img, labels, sample_weights=None):
        text_repr = self.text_model(texts, img, labels, sample_weights)
        text_repr = text_repr.transpose(1, 2)
        text_repr = self.text_channel_reducer(text_repr).mean(2).unsqueeze(2)
        text_repr = text_repr.expand((*text_repr.size()[:-1], self.imf_width)).unsqueeze(3)
        text_repr = text_repr.expand((*text_repr.size()[:-1], self.imf_width))

        image_repr = self.im_model(img)
        repr = torch.cat((image_repr, text_repr), dim=1)
        vectors = self.featurizer(repr)
        logits = self.classifier(vectors).squeeze()

        loss = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
        preds = logits.max(dim=1).indices
        logits = torch.softmax(logits, dim=1)

        return logits, preds, vectors.mean(1), vectors, loss

