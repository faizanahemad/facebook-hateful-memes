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
import torchvision
import os
warnings.filterwarnings('ignore')

from facebook_hateful_memes_detector.utils.globals import set_global, get_global

set_global("cache_dir", os.path.join(os.getcwd(), "cache"))
set_global("dataloader_workers", 0)
set_global("use_autocast", False)

from facebook_hateful_memes_detector.utils import read_json_lines_into_df, in_notebook, set_device

get_global("cache_dir")
from facebook_hateful_memes_detector.models import Fasttext1DCNNModel, MultiImageMultiTextAttentionEarlyFusionModel, VilBertVisualBertModel
from facebook_hateful_memes_detector.preprocessing import TextImageDataset, my_collate, get_datasets, get_image2torchvision_transforms, TextAugment
from facebook_hateful_memes_detector.preprocessing import DefinedRotation, QuadrantCut, ImageAugment, DefinedAffine, DefinedColorJitter, \
    DefinedRandomPerspective
from facebook_hateful_memes_detector.training import *
import facebook_hateful_memes_detector

reload(facebook_hateful_memes_detector)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)

choice_probas = {"keyboard": 0.1, "char_substitute": 0.0, "char_insert": 0.1, "char_swap": 0.0, "ocr": 0.0, "char_delete": 0.1,
                 "fasttext": 0.0, "glove_twitter": 0.0, "glove_wiki": 0.0, "word2vec": 0.0, "split": 0.1,
                 "stopword_insert": 0.3, "word_join": 0.1, "word_cutout": 0.8,
                 "text_rotate": 0.5, "sentence_shuffle": 0.5, "one_third_cut": 0.4, "half_cut": 0.1}
preprocess_text = TextAugment([1.0, 0.0], choice_probas, fasttext_file="wiki-news-300d-1M-subword.bin")

im_transform = ImageAugment(count_proba=[0.1, 0.9],
                            augs_dict=dict(grayscale=transforms.Grayscale(num_output_channels=3),
                                           hflip=transforms.RandomHorizontalFlip(p=1.0),
                                           rc2=transforms.Compose([transforms.Resize(480), transforms.CenterCrop(400)]),
                                           rotate=DefinedRotation(15),
                                           affine=DefinedAffine(0, scale=(0.6, 0.6)),
                                           ),
                            choice_probas="uniform"
                            )

data = get_datasets(data_dir="../data/", train_text_transform=preprocess_text, train_image_transform=im_transform,
                    test_text_transform=None, test_image_transform=None,
                    cache_images=False, use_images=True, dev=False,
                    keep_original_text=False, keep_original_image=False,
                    keep_processed_image=True, keep_torchvision_image=False,)


data["test"] = data["dev"].tail(256).head(128)

adam = torch.optim.Adam
adam_params = params = dict(lr=1e-3, weight_decay=1e-7)
optimizer = adam
optimizer_params = adam_params

batch_size=16

lr_strategy = {
    "vilbert": {
        "lr": optimizer_params["lr"] / 10000,
        "model": {
            "bert": {
                "t_pooler": {
                    "lr": optimizer_params["lr"],
                    "finetune": True
                },
                "v_pooler": {
                    "lr": optimizer_params["lr"],
                    "finetune": True
                }
            },
            "classifier": {
                "lr": optimizer_params["lr"],
                "finetune": True
            }
        },
        "finetune": False
    }
}

lr_strategy = {
    "visual_bert": {
        "model": {
            "bert": {
                "pooler": {
                    "lr": optimizer_params["lr"] / 1e1,
                    "finetune": True
                }
            }
        },
        "finetune": False
    },
}

model_fn = model_builder(
    VilBertVisualBertModel,
    dict(
        model_name={"visual_bert": dict(gaussian_noise=0.0, dropout=0.0)},
        num_classes=2,
        gaussian_noise=0.0,
        dropout=0.0,
        word_masking_proba=0.0,
        featurizer="pass",
        final_layer_builder=fb_1d_loss_builder,
        internal_dims=256,
        classifier_dims=768,
        n_tokens_in=96,
        n_tokens_out=16,
        n_layers=2,
        loss="classification",
        dice_loss_coef=0.0,
        auc_loss_coef=0.0,
    ),
    per_param_opts_fn=lr_strategy,
    optimiser_class=optimizer,
    optimiser_params=optimizer_params)

model, _ = model_fn()
sf, _ = predict(model, data, batch_size)

print(sf.head())

from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

labels_list = data["test"].label
proba_list = sf.proba
predictions_list = sf.label

auc = roc_auc_score(labels_list, proba_list, multi_class="ovo", average="macro")
# p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(labels_list, predictions_list, average="micro")
prfs = precision_recall_fscore_support(labels_list, predictions_list, average=None, labels=[0, 1])
map = average_precision_score(labels_list, proba_list)
acc = accuracy_score(labels_list, predictions_list)
validation_scores = [map, acc, auc]
print("scores = ", dict(zip(["map", "acc", "auc"], ["%.4f" % v for v in validation_scores])))
