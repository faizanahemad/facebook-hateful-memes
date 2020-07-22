


import pandas as pd
import numpy as np
import jsonlines
import seaborn as sns
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from importlib import reload
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', '{:0.3f}'.format)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings
import os
import torchvision
warnings.filterwarnings('ignore')

from facebook_hateful_memes_detector.utils.globals import set_global, get_global
set_global("cache_dir", os.path.join(os.getcwd(), "cache"))
set_global("dataloader_workers", 0)
set_global("use_autocast", True)
set_global("models_dir", os.path.join(os.getcwd(), "cache"))

from facebook_hateful_memes_detector.utils import read_json_lines_into_df, in_notebook, set_device
get_global("cache_dir")
from facebook_hateful_memes_detector.models import Fasttext1DCNNModel, MultiImageMultiTextAttentionEarlyFusionModel, LangFeaturesModel, AlbertClassifer, TransformerImageModel

from facebook_hateful_memes_detector.preprocessing import TextImageDataset, my_collate, get_datasets, get_image2torchvision_transforms, TextAugment
from facebook_hateful_memes_detector.preprocessing import DefinedRotation, QuadrantCut, ImageAugment, DefinedAffine
from facebook_hateful_memes_detector.training import *
import facebook_hateful_memes_detector
reload(facebook_hateful_memes_detector)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)


data = get_datasets(data_dir="../data/", train_text_transform=None, train_image_transform=None,
                    test_text_transform=None, test_image_transform=None,
                    train_torchvision_image_transform=transforms.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),
                    test_torchvision_image_transform=None,
                    cache_images = True, use_images = True, dev=True, test_dev=True,
                    keep_original_text=False, keep_original_image=False,
                    keep_processed_image=True, keep_torchvision_image=True,)

adam = torch.optim.Adam
adam_params = params=dict(lr=1e-4, weight_decay=1e-6)
optimizer = adam
optimizer_params = adam_params

lr_strategy = {
    "im_models": {
        "lr": optimizer_params["lr"] / 10,
        "torchvision_resnet18_ssl-contrastive": {
            "lambd": {
                "8": {
                    "finetune": True
                }
            },
            "lr": optimizer_params["lr"] / 10,
            "finetune": False,
        },
        "vgg_face": {
            "lr": optimizer_params["lr"] / 10,
            "lambd": {
                "0": {
                    "feat_extract": {
                        "finetune": True
                    }
                }
            },
            "finetune": False,
        }
    }
}


model_fn = model_builder(
    TransformerImageModel,
    dict(
        image_models=[
            #             {
            #                 "model": 'caption_features',
            #                 "gaussian_noise": 0.0,
            #                 "dropout": 0.0
            #             },
            {
                "model": 'vgg_face',
                "gaussian_noise": 0.0,
                "dropout": 0.0,
            },
            #             {
            #                 "model": 'detr_resnet50',
            #                 "gaussian_noise": 0.0,
            #                 "dropout": 0.0
            #             },
            #             {
            #                 "model": 'detr_resnet50_panoptic',
            #                 "gaussian_noise": 0.0,
            #                 "dropout": 0.0
            #             },
            {
                "model": "torchvision_resnet18_ssl-contrastive",
                "large_rf": True,
                "dropout": 0.0,
                "gaussian_noise": 0.0,
            },
        ],
        classifier_dims=256,
        num_classes=2,
        gaussian_noise=0.1,
        dropout=0.2,
        word_masking_proba=0.15,
        internal_dims=512,
        final_layer_builder=fb_1d_loss_builder,
        n_layers=2,
        n_encoders=3,
        n_decoders=3,
        n_tokens_in=96,
        n_tokens_out=32,
        featurizer="transformer",
        model='allenai/longformer-base-4096',
        loss="focal",
        classification_head="decoder_ensemble",  # decoder_ensemble
        dice_loss_coef=0.0,
        auc_loss_coef=0.5,
        finetune=False,
    ),
    per_param_opts_fn=lr_strategy,
    optimiser_class=optimizer,
    optimiser_params=optimizer_params)

batch_size=8
epochs = 10
kfold = False
results, prfs = train_validate_ntimes(
    model_fn,
    data,
    batch_size,
    epochs,
    kfold=kfold,
    scheduler_init_fn=None,
    model_call_back=None, # reg_sched
    validation_epochs=[4, 7, 9, 11, 14, 17, 19, 23, 27, 31, 34, 37, 41, 44, 47, 51, 54],
    show_model_stats=False,
    sampling_policy="without_replacement")
r2, p2 = results, prfs
results
prfs
