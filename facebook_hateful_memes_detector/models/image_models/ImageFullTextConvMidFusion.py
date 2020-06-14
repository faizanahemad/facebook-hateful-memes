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

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_torchvision_classification_models
from ..classifiers import CNN1DClassifier, GRUClassifier, TransformerClassifier, TransformerEnsembleClassifier
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel


class ImageFullTextConvMidFusionModel(nn.Module):
    def __init__(self, image_model, num_classes,
                 text_model_class, text_model_params, text_in_channels,
                 finetune_image_model=False,
                 gaussian_noise=0.0, dropout=0.0,
                 internal_dims=128, classifier_dims=128,
                 **kwargs):
        super(ImageFullTextConvMidFusionModel, self).__init__()
        if type(image_model) == str:
            if "torchvision" in image_model:
                net = image_model.split("_")[-1]
                im_model, im_shape = get_torchvision_classification_models(net, finetune_image_model)
            else:
                raise NotImplementedError(image_model)
            self.im_model = im_model
            self.imf_width = im_shape[-1]
            l1 = nn.Conv2d(im_shape[0], int(internal_dims / 2), 1, 1, padding=0, groups=1, bias=False)
            init_fc(l1, "leaky_relu")
            l2 = nn.Conv2d(int(internal_dims / 2), internal_dims, 3, 1, padding=1, groups=1, bias=False)
            init_fc(l2, "leaky_relu")
            self.im_proc = nn.Sequential(nn.Dropout(dropout), l1, nn.LeakyReLU(),
                                         GaussianNoise(gaussian_noise), l2, nn.LeakyReLU())

        self.text_model = text_model_class(**text_model_params)
        # Normalize on 2nd dim for both text and img
        l2 = nn.Conv2d(internal_dims + text_in_channels, int((internal_dims + text_in_channels)/2), 1, 1, padding=0, groups=1, bias=False)
        init_fc(l2, "leaky_relu")
        l3 = nn.Conv2d(int((internal_dims + text_in_channels)/2), internal_dims, 3, 1, padding=4, groups=1, bias=False)
        init_fc(l3, "leaky_relu")
        l4 = nn.Conv2d(internal_dims, classifier_dims, 3, 1, padding=0, groups=1, bias=False)
        init_fc(l4, "linear")
        self.featurizer = nn.Sequential(nn.Dropout(dropout), l2, nn.LeakyReLU(), GaussianNoise(gaussian_noise), l3,
                                        nn.LeakyReLU(), nn.Dropout(dropout), l4) # 5x5 for resnet18 from 7x7

        l2 = nn.Conv2d(classifier_dims, classifier_dims, 3, 1, padding=0, groups=1, bias=False)
        init_fc(l2, "leaky_relu")
        l3 = nn.Conv2d(classifier_dims, num_classes, 3, 1, padding=0, groups=1, bias=False)
        init_fc(l3, "linear")
        self.classifier = nn.Sequential(nn.LeakyReLU(), nn.Dropout(dropout), l2, nn.LeakyReLU(), l3, nn.AdaptiveAvgPool2d(1))
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.finetune_image_model = finetune_image_model

    def forward(self, texts: List[str], img, labels, sample_weights=None):
        _, _, _, text_repr, _ = self.text_model(texts, img, labels, sample_weights)
        text_repr = text_repr.mean(1).unsqueeze(2)
        text_repr = text_repr.expand((*text_repr.size()[:-1], self.imf_width)).unsqueeze(3)
        text_repr = text_repr.expand((*text_repr.size()[:-1], self.imf_width))

        if not self.finetune_image_model:
            with torch.no_grad():
                image_repr = self.im_model(img)
        else:
            image_repr = self.im_model(img)
        image_repr = self.im_proc(image_repr)
        repr = torch.cat((image_repr, text_repr), dim=1)
        vectors = self.featurizer(repr)
        logits = self.classifier(vectors).squeeze()

        loss = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
        preds = logits.max(dim=1).indices
        logits = torch.softmax(logits, dim=1)

        return logits, preds, vectors.mean(1), vectors, loss

