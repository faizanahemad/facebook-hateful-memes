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


class MultiImageMultiTextAttentionEarlyFusionModel(nn.Module):
    def __init__(self, image_models, num_classes,
                 text_models,
                 gaussian_noise, dropout,
                 internal_dims, classifier_dims,
                 final_layer_builder,
                 n_tokens_out, n_layers,
                 **kwargs):
        super(MultiImageMultiTextAttentionEarlyFusionModel, self).__init__()
        assert type(image_models) == list
        names, im_models, im_shapes, im_procs, im_finetune = [], [], [], [], []
        for imo in image_models:
            if type(imo) == list or type(imo) == tuple:
                finetune = imo[2]
                large_rf = imo[1]
                imo = imo[0]
            elif type(imo) == dict:
                finetune = imo["finetune"]
                large_rf = imo["large_rf"]
                imo = imo["model"]
            elif type(imo) == str:
                finetune = False
                large_rf = True
            else:
                raise NotImplementedError()

            if "torchvision" in imo:
                net = imo.split("_")[-1]
                im_model, im_shape = get_torchvision_classification_models(net, large_rf, finetune)

                l1 = nn.Conv2d(im_shape[0], im_shape[0], 3, 1, padding=1, groups=16, bias=False)
                init_fc(l1, "leaky_relu")
                im_proc = nn.Sequential(nn.Dropout(dropout), l1, nn.LeakyReLU())

            elif imo == "caption_features":
                im_model = LambdaLayer(get_image_info_fn(enable_encoder_feats=True)["get_batch_encoder_feats"])
                im_shape = (512, 100)
                im_proc = nn.Identity()

            elif imo == "faster_rcnn":
                im_shape = (2048, 100)
                im_model = LambdaLayer(get_image_info_fn(enable_encoder_feats=False)["get_batch_img_roi_features"])
                im_proc = nn.Identity()

            elif imo == "lxmert_faster_rcnn":
                im_shape = (2048, 36)
                im_model = LambdaLayer(get_image_info_fn(enable_encoder_feats=False)["get_batch_lxmert_roi_features"])
                im_proc = nn.Identity()
            elif "detr" in imo:
                im_shape = (256, 100)
                im_model = LambdaLayer(get_detr_model(get_device(), imo)["batch_detr_fn"])
                im_proc = nn.Identity()
            else:
                raise NotImplementedError(imo)

            names.append(imo)
            im_finetune.append(finetune)
            im_models.append(im_model)
            im_shapes.append(im_shape)
            im_procs.append(im_proc)
        self.im_models = nn.ModuleDict(dict(zip(names, im_models)))
        self.im_procs = nn.ModuleDict(dict(zip(names, im_procs)))
        self.im_finetune = dict(zip(names, im_finetune))
        self.im_shapes = dict(zip(names, im_shapes))
        self.require_raw_img = {"detr_demo", 'detr_resnet50', 'detr_resnet50_panoptic', 'detr_resnet101', 'detr_resnet101_panoptic',
                                "ssd", "faster_rcnn", "lxmert_faster_rcnn", "caption_features"}

        assert type(text_models) == list
        tx_models, tx_names = [], []
        for i, tm in enumerate(text_models):
            assert type(tm) == dict
            text_model_class, text_model_params, text_in_channels, text_in_tokens = tm["cls"], tm["params"], tm["in_channels"], tm["in_tokens"]
            text_model = text_model_class(**text_model_params)
            tx_models.append(text_model)
            tx_names.append("tx_" + str(i))

        self.tx_models = nn.ModuleDict(dict(zip(tx_names, tx_models)))
        self.text_models = dict(zip(tx_names, text_models))

        ensemble_conf = {k: dict(is2d=len(v) == 3, n_tokens_in=(v[-1] * v[-1]) if len(v) == 3 else v[-1], n_channels_in=v[0]) for k, v in self.im_shapes.items()}
        text_ensemble_conf = {k: dict(is2d=False, n_tokens_in=v["in_tokens"], n_channels_in=v["in_channels"]) for k, v in
             self.text_models.items()}
        ensemble_conf.update(text_ensemble_conf)
        # ensemble_conf = text_ensemble_conf
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
        for k, m in self.tx_models.items():
            _, _, text_repr, _ = m(sampleList)
            vectors[k] = text_repr.to(get_device())
            clean_memory()

        del sampleList

        if self.require_raw_img.isdisjoint(set(self.im_models.keys())):
            del image

        if len(set(self.im_models.keys()) - self.require_raw_img) > 0:
            img = img.to(get_device())
        else:
            del img

        clean_memory()
        for k, m in self.im_models.items():
            if self.im_finetune[k]:
                im_repr = m(image if k in self.require_raw_img else img)
            else:
                with torch.no_grad():
                    im_repr = m(image if k in self.require_raw_img else img)
            im_repr = self.im_procs[k](im_repr)
            vectors[k] = im_repr.to(get_device())
            clean_memory()

        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels)
        return logits, vectors.mean(1), vectors, loss
