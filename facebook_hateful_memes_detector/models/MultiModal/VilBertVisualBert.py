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

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_torchvision_classification_models, get_device
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer, TransformerEnsembleFeaturizer
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel
from ..external.mmf import get_vilbert, get_visual_bert


class VilBertVisualBertModel(nn.Module):
    def __init__(self, model_name, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, classifier_dims,
                 final_layer_builder,
                 n_tokens_out, n_layers,
                 finetune=False,
                 **kwargs):
        super(VilBertVisualBertModel, self).__init__()

        if model_name == "vilbert":
            m = get_vilbert(get_device())
        elif model_name == "visual_bert":
            m = get_visual_bert(get_device())
        else:
            raise NotImplementedError()
        self.model, self.tknzr = m["model"], m["tokenizer"]
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False


        # ensemble_conf = text_ensemble_conf
        self.featurizer = TransformerEnsembleFeaturizer(ensemble_conf, n_tokens_out, classifier_dims, internal_dims,
                                                        n_layers, gaussian_noise, dropout)

        self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, )
        self.finetune = finetune

    def forward(self, sampleList: SampleList):
        texts = sampleList.text
        img = sampleList.torchvision_image
        orig_image = sampleList.original_image
        labels = sampleList.label
        sample_weights = sampleList.sample_weight

        vectors = dict()
        for k, m in self.tx_models.items():
            _, _, text_repr, _ = m(sampleList)
            vectors[k] = text_repr

        for k, m in self.im_models.items():
            if self.finetune_image_model:
                im_repr = m(img)
            else:
                with torch.no_grad():
                    im_repr = m(img)
            im_repr = self.im_procs[k](im_repr)
            vectors[k] = im_repr

        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels)
        return logits, vectors.mean(1), vectors, loss
