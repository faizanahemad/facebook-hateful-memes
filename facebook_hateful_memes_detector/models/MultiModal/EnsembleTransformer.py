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

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_torchvision_classification_models, get_image_info_fn, LambdaLayer, get_device, \
    dict2sampleList, clean_memory
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer, TransformerEnsembleFeaturizer
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel
from ..external.detr import get_detr_model


class EnsembleTransformerModel(nn.Module):
    def __init__(self, models, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, classifier_dims,
                 final_layer_builder,
                 n_tokens_out, n_layers,
                 **kwargs):
        super(EnsembleTransformerModel, self).__init__()
        assert type(models) == list
        names, im_models, im_shapes, im_procs, im_finetune = [], [], [], [], []
        for i, imo in enumerate(models):
            if type(imo) == dict:
                finetune = imo["finetune"]
                params = imo["params"]
                mdl_shape = imo["shape"]  # Embedding Dim, SeqLen
                mdl = imo["model"] if "model" in imo else None
                mdl_class = imo["model_class"] if "model_class" in imo else None

                if mdl is None:
                    assert mdl_class is not None
                    mdl = mdl_class(**params)
                    for p in mdl.parameters():
                        p.requires_grad = finetune

            else:
                raise NotImplementedError()

            names.append(str(i))
            im_finetune.append(finetune)
            im_models.append(mdl)
            im_shapes.append(mdl_shape)
        self.im_models = nn.ModuleDict(dict(zip(names, im_models)))
        self.im_finetune = dict(zip(names, im_finetune))
        self.im_shapes = dict(zip(names, im_shapes))

        ensemble_conf = {k: dict(is2d=len(v) == 3, n_tokens_in=(v[-1] * v[-1]) if len(v) == 3 else v[-1], n_channels_in=v[0]) for k, v in self.im_shapes.items()}
        self.featurizer = TransformerEnsembleFeaturizer(ensemble_conf, n_tokens_out, classifier_dims, internal_dims,
                                                        n_layers, gaussian_noise, dropout)

        self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, )

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        img = sampleList.torchvision_image
        image = sampleList.image
        labels = torch.tensor(sampleList.label, dtype=float).to(get_device())
        sample_weights = sampleList.sample_weight

        vectors = dict()
        for k, m in self.im_models.items():
            _, _, repr, _ = m(sampleList)
            vectors[k] = repr.to(get_device())
            clean_memory()

        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels)
        return logits, vectors.mean(1), vectors, loss
