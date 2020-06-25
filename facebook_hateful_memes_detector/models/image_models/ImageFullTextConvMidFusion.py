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
from mmf.common import SampleList
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_torchvision_classification_models, dict2sampleList, get_device, clean_memory
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer, TransformerEnsembleFeaturizer
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel


class ImageFullTextConvMidFusionModel(nn.Module):
    def __init__(self, image_model, num_classes,
                 text_model_class, text_model_params, text_in_channels,
                 gaussian_noise, dropout,
                 internal_dims, classifier_dims,
                 final_layer_builder,
                 large_rf,
                 finetune_image_model=False,
                 **kwargs):
        super(ImageFullTextConvMidFusionModel, self).__init__()
        if type(image_model) == str:
            if "torchvision" in image_model:
                net = image_model.split("_")[-1]
                im_model, im_shape = get_torchvision_classification_models(net, large_rf, finetune_image_model)
            else:
                raise NotImplementedError(image_model)
            self.im_model = im_model
            self.imf_width = im_shape[-1]
            l1 = nn.Conv2d(im_shape[0], internal_dims * 2, 1, 1, padding=0, groups=4, bias=False)
            init_fc(l1, "leaky_relu")
            l2 = nn.Conv2d(internal_dims * 2, internal_dims, 1, 1, padding=0, groups=1, bias=False)
            init_fc(l2, "leaky_relu")
            self.im_proc = nn.Sequential(nn.Dropout(dropout), l1, nn.LeakyReLU(),
                                         GaussianNoise(gaussian_noise), l2, nn.LeakyReLU())

        self.text_model = text_model_class(**text_model_params)
        # Normalize on 2nd dim for both text and img
        l2 = nn.Conv2d(internal_dims + text_in_channels, int((internal_dims + text_in_channels) * 2), 1, 1, padding=0, groups=1, bias=False)
        init_fc(l2, "leaky_relu")
        l3 = nn.Conv2d(int((internal_dims + text_in_channels) * 2), internal_dims, 1, 1, padding=0, groups=1, bias=False)
        init_fc(l3, "leaky_relu")
        l4 = nn.Conv2d(internal_dims, classifier_dims, 3, 1, padding=0, groups=1, bias=False)
        init_fc(l4, "leaky_relu")
        self.featurizer = nn.Sequential(nn.Dropout(dropout), l2, nn.LeakyReLU(), GaussianNoise(gaussian_noise), l3,
                                        nn.LeakyReLU(), nn.Dropout(dropout), l4, nn.LeakyReLU())  # 5x5 for resnet18 from 7x7

        self.final_layer = final_layer_builder(classifier_dims, (self.imf_width - 2) ** 2, num_classes, dropout, )
        self.num_classes = num_classes
        self.finetune_image_model = finetune_image_model
        self.text_layer_norm = nn.LayerNorm(text_in_channels)
        self.im_layer_norm = nn.LayerNorm(internal_dims)
        self.internal_dims = internal_dims

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        img = sampleList.torchvision_image.to(get_device())
        labels = torch.tensor(sampleList.label, dtype=float).to(get_device())
        sample_weights = sampleList.sample_weight
        _, _, text_repr, _ = self.text_model(sampleList)
        del sampleList
        clean_memory()
        text_repr = self.text_layer_norm(text_repr)
        text_repr = text_repr.mean(1).unsqueeze(2)
        text_repr = text_repr.expand((*text_repr.size()[:-1], self.imf_width)).unsqueeze(3)
        text_repr = text_repr.expand((*text_repr.size()[:-1], self.imf_width))

        if not self.finetune_image_model:
            with torch.no_grad():
                image_repr = self.im_model(img)
        else:
            image_repr = self.im_model(img)
        del img
        image_repr = self.im_proc(image_repr)
        image_repr = self.im_layer_norm(image_repr.transpose(1, 2).transpose(2, 3)).transpose(2, 3).transpose(1, 2)
        repr = torch.cat((image_repr, text_repr), dim=1)
        del image_repr
        del text_repr
        clean_memory()
        vectors = self.featurizer(repr)
        vectors = vectors.transpose(1, 2).transpose(2, 3).flatten(1, 2)  # N,C,H,W -> N,HxW,C
        logits, loss = self.final_layer(vectors, labels)
        return logits, vectors.mean(1), vectors, loss

