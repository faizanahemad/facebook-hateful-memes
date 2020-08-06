import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmf.common import SampleList, Sample
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import contractions
import pandas as pd
import jsonlines
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_PADDING_INDEX
from spacy import glossary
from .globals import get_device, set_device, set_cpu_as_device, set_first_gpu, memory, build_cache, get_global
import os
import torch
import gc
import os
import random
from typing import Optional, Any
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, LayerNorm, TransformerEncoderLayer, CrossEntropyLoss
from torch.nn.init import xavier_uniform_
import math
import copy
from sklearn.metrics import accuracy_score

from ..models.classifiers import TransformerFeaturizer
from ..utils import *
DIR = os.path.dirname(os.path.realpath(__file__))


# TODO: Use both VggFace and Resnet50-SSL 7x7, 3x3, global view -> 61
# TODO: Retrain this for AugSim, Differentiator, SimCLR objective,
# TODO: Enable Multi-view (HFlip, Zoom-in, Zoom-out, +15, -15 Rotate), Add 2 layer encoder self-attn take first 128/64 tokens
# TODO: Linear layer 2048->768 before encoder
# Don't Multi-View now.


#

class ImageModelShim(nn.Module):
    def __init__(self, resnet="resnet50_ssl", n_tokens=64, out_channels=768, n_encoders=2, dropout=0.0, gaussian_noise=0.0, **kwargs):
        super().__init__()
        resnet_model, resnet_shape = get_torchvision_classification_models(resnet, True)
        vgg_shape = (256, 1)
        vgg_model = get_vgg_face_model()

        self.resnet_model = resnet_model
        self.vgg_model = vgg_model
        self.resnet = resnet

        gaussian_noise = GaussianNoise(gaussian_noise)

        lin = nn.Linear(resnet_shape[0], out_channels)
        init_fc(lin, "linear")
        self.resnet_reshape = nn.Sequential(nn.Dropout(dropout), lin, gaussian_noise)

        lin = nn.Linear(vgg_shape[0], out_channels)
        init_fc(lin, "linear")
        self.vgg_reshape = nn.Sequential(nn.Dropout(dropout), lin, gaussian_noise)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        half_dim = 3
        self.half_pool = nn.AdaptiveMaxPool2d(half_dim)

        n_tokens_in = (resnet_shape[1] * resnet_shape[1]) + 1 + (half_dim * half_dim) + 1
        featurizer = TransformerFeaturizer(n_tokens_in, out_channels, n_tokens,
                                           out_channels,
                                           out_channels, n_encoders, 0,
                                           gaussian_noise, dropout, self.attention_drop_proba)
        self.featurizer = featurizer

        if "stored_model" in kwargs and kwargs["stored_model"] is not None:
            load_stored_params(self, kwargs["stored_model"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        resnet_in = self.resnet_model(images)
        resnet_lrf = self.half_pool(resnet_in)
        resnet_global = self.global_pool(resnet_in).squeeze().unsqueeze(1)
        vgg_face_in = self.vgg_reshape(self.vgg_model(images).squeeze().unsqueeze(1))

        resnet_in = resnet_in.flatten(1, 2).transpose(1, 2)  # B,C,H,W -> B,HxW,C
        resnet_lrf = resnet_lrf.flatten(1, 2).transpose(1, 2)

        resnet_out = self.resnet_reshape(torch.cat([resnet_in, resnet_lrf, resnet_global], 1))

        seq = torch.cat([resnet_out, vgg_face_in], 1)
        seq = self.featurizer(seq)
        return seq


class ImageCaptioningShim(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()
        lin0 = nn.Linear(512, 512)
        init_fc(lin0, "leaky_relu")
        lin = nn.Linear(512, 768)
        init_fc(lin, "linear")
        self.reshape = nn.Sequential(nn.Dropout(dropout), lin0, nn.LeakyReLU(), nn.Dropout(dropout), lin, nn.LayerNorm(768))
        self.captioner = get_image_info_fn(enable_encoder_feats=True)["get_batch_encoder_feats"]

        if "stored_model" in kwargs and kwargs["stored_model"] is not None:
            load_stored_params(self, kwargs["stored_model"])

    def forward(self, images: List, ignore_cache: List[bool] = None):
        caption_features = self.captioner(images, ignore_cache)
        caption_features = self.reshape(caption_features)
        return caption_features




