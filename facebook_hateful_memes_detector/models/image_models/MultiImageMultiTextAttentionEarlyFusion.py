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
    dict2sampleList, clean_memory, get_vgg_face_model
from ..classifiers import TransformerEnsembleFeaturizer
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel
from ..external.detr import get_detr_model


class MultiImageMultiTextAttentionEarlyFusionModel(nn.Module):
    def __init__(self, image_models, num_classes,
                 text_models,
                 gaussian_noise, dropout,
                 internal_dims, classifier_dims,
                 final_layer_builder,
                 n_tokens_out, n_encoders, n_decoders,
                 **kwargs):
        super(MultiImageMultiTextAttentionEarlyFusionModel, self).__init__()
        assert type(image_models) == list
        names, im_models, im_shapes, im_procs, im_finetune = [], [], [], [], []
        for imo in image_models:
            if type(imo) == dict:
                module_gaussian = imo["gaussian_noise"] if "gaussian_noise" in imo else 0.0
                module_dropout = imo["dropout"] if "dropout" in imo else 0.0
                finetune = imo["finetune"] if "finetune" in imo else False
                large_rf = imo["large_rf"] if "large_rf" in imo else True
                imo = imo["model"]
            elif type(imo) == str:
                module_gaussian = 0.0
                module_dropout = 0.0
                finetune = False
                large_rf = True
            else:
                raise NotImplementedError()

            if "torchvision" in imo:
                net = "_".join(imo.split("_")[1:])
                im_model, im_shape = get_torchvision_classification_models(net, large_rf, finetune)
                im_model = LambdaLayer(im_model, module_gaussian, module_dropout)
                im_proc = nn.Identity()

            elif imo == "caption_features":
                im_model = LambdaLayer(get_image_info_fn(enable_encoder_feats=True)["get_batch_encoder_feats"], module_gaussian, module_dropout)
                im_shape = (512, 100)
                im_proc = nn.Identity()

            elif imo == "faster_rcnn":
                im_shape = (2048, 100)
                im_model = LambdaLayer(get_image_info_fn(enable_encoder_feats=False)["get_batch_img_roi_features"], module_gaussian, module_dropout)
                im_proc = nn.Identity()

            elif imo == "lxmert_faster_rcnn":
                im_shape = (2048, 36)
                im_model = LambdaLayer(get_image_info_fn(enable_encoder_feats=False)["get_batch_lxmert_roi_features"], module_gaussian, module_dropout)
                im_proc = nn.Identity()
            elif "detr" in imo:
                im_shape = (256, 100)
                im_model = LambdaLayer(get_detr_model(get_device(), imo)["batch_detr_fn"], module_gaussian, module_dropout)
                im_proc = nn.Identity()
            elif "vgg_face" in imo:
                im_shape = (256, 1)
                im_model = LambdaLayer(get_vgg_face_model(), module_gaussian, module_dropout)
                im_proc = nn.Identity()
            else:
                raise NotImplementedError(imo)

            names.append(imo)
            im_finetune.append(finetune)
            im_models.append(im_model)
            im_shapes.append(im_shape)
            im_procs.append(im_proc)
        self.im_models = nn.ModuleDict(dict(zip(names, im_models)))
        self.post_procs = nn.ModuleDict(dict(zip(names, im_procs)))
        self.im_finetune = dict(zip(names, im_finetune))
        self.im_shapes = dict(zip(names, im_shapes))
        self.require_raw_img = {"detr_demo", 'detr_resnet50', 'detr_resnet50_panoptic', 'detr_resnet101', 'detr_resnet101_panoptic',
                                "ssd", "faster_rcnn", "lxmert_faster_rcnn", "caption_features"}

        assert type(text_models) == list
        tx_models, tx_names, tx_methods = [], [], []
        for i, tm in enumerate(text_models):
            assert type(tm) == dict
            text_model_class, text_model_params, text_in_channels, text_in_tokens = tm["cls"], tm["params"], tm["in_channels"], tm["in_tokens"]
            text_fwd = tm["forward"] if "forward" in tm else "__call__"
            assert text_fwd in ["__call__", "get_word_vectors"]
            text_model = text_model_class(**text_model_params)
            if text_fwd != "__call__":
                if hasattr(text_model, "featurizer"):
                    text_model.featurizer = None
                    del text_model.featurizer
                if hasattr(text_model, "final_layer"):
                    text_model.final_layer = None
                    del text_model.final_layer
            module_gaussian = tm["gaussian_noise"] if "gaussian_noise" in tm else 0.0
            module_dropout = tm["dropout"] if "dropout" in tm else 0.0
            tx_models.append(text_model)
            self.post_procs["tx_" + str(i)] = nn.Sequential(nn.Dropout(module_dropout), GaussianNoise(module_gaussian))
            tx_names.append("tx_" + str(i))
            tx_methods.append(text_fwd)

        self.tx_models = nn.ModuleDict(dict(zip(tx_names, tx_models)))
        self.text_models = dict(zip(tx_names, text_models))
        self.tx_methods = dict(zip(tx_names, tx_methods))

        ensemble_conf = {k: dict(is2d=len(v) == 3, n_tokens_in=(v[-1] * v[-1]) if len(v) == 3 else v[-1], n_channels_in=v[0]) for k, v in self.im_shapes.items()}
        text_ensemble_conf = {k: dict(is2d=False, n_tokens_in=v["in_tokens"], n_channels_in=v["in_channels"]) for k, v in
             self.text_models.items()}
        ensemble_conf.update(text_ensemble_conf)
        # ensemble_conf = text_ensemble_conf
        self.featurizer = TransformerEnsembleFeaturizer(ensemble_conf, n_tokens_out, classifier_dims, internal_dims,
                                                        n_encoders, n_decoders, gaussian_noise, dropout)

        self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, )

    def get_vectors(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        img = sampleList.torchvision_image
        image = sampleList.image
        sample_weights = sampleList.sample_weight

        vectors = dict()
        for k, m in self.tx_models.items():
            r = getattr(m, self.tx_methods[k])(sampleList if self.tx_methods[k] == "__call__" else sampleList.text)
            text_repr = r[2] if self.tx_methods[k] == "__call__" else r
            text_repr = self.post_procs[k](text_repr)
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
            im_repr = self.post_procs[k](im_repr)
            vectors[k] = im_repr.to(get_device())
            clean_memory()

        vectors = self.featurizer(vectors)
        return vectors

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        labels = torch.tensor(sampleList.label, dtype=float).to(get_device())
        vectors = self.get_vectors(sampleList)
        del sampleList
        clean_memory()
        logits, loss = self.final_layer(vectors, labels)
        return logits, vectors.mean(1), vectors, loss
