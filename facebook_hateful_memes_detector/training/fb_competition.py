from typing import List, Dict

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import contractions
import pandas as pd

from ..utils import in_notebook, CNNHead, AveragedLinearHead, OneTokenPositionLinearHead, MultiTaskForward, CNN2DHead
from ..preprocessing import my_collate, make_weights_for_balanced_classes, TextImageDataset
import gc
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset
from transformers import optimization
from .generic import *


def fb_1d_loss_builder(n_dims, n_tokens, n_out, dropout,):
    cnn = CNNHead(n_dims, n_tokens, n_out, dropout, "classification")
    mtf = MultiTaskForward([cnn])
    return mtf


def fb_2d_loss_builder(n_dims, n_out, dropout,):
    cnn = CNN2DHead(n_dims, n_out, dropout, "classification")
    mtf = MultiTaskForward([cnn])
    return mtf


def train_and_predict(model_fn, datadict, batch_size, epochs, augmentation_weights: Dict[str, float],
                      multi_eval=False, scheduler_init_fn=None):
    train_df = datadict["train"]
    train_df["augmented"] = False
    train_df["augment_type"] = "None"
    metadata = datadict["metadata"]
    augmented_data = metadata["augmented_data"]
    train_augmented = datadict["train_augmented"] if augmented_data else train_df
    train_augmented["sample_weights"] = 0.0
    for k, v in augmentation_weights.items():
        train_augmented.loc[train_augmented["augment_type"] == k, "sample_weights"] = v
    train_augmented = train_augmented[train_augmented["sample_weights"] > 0]
    dataset = convert_dataframe_to_dataset(train_augmented, metadata, True)

    model, optimizer = model_fn(dataset=dataset)
    train_losses, learning_rates = train(model, optimizer, scheduler_init_fn, batch_size, epochs, dataset, plot=True)
    test = datadict["test"]
    test["augmented"] = False
    test["augment_type"] = "None"
    test_augmented: pd.DataFrame = datadict["test_augmented"] if augmented_data else test

    if not multi_eval:
        test_augmented = test_augmented[~test_augmented["augmented"]]
        test_augmented["sample_weights"] = 1.0
    else:
        test_augmented["sample_weights"] = 0.0
        for k, v in augmentation_weights.items():
            test_augmented[test_augmented["augment_type"] == k, "sample_weights"] = v

    test_dataset = convert_dataframe_to_dataset(test_augmented, metadata, False)
    proba_list, predictions_list, labels_list = generate_predictions(model, batch_size, test_dataset)
    test_augmented["proba"] = proba_list
    test_augmented["predictions_list"] = predictions_list
    test_augmented["weighted_proba"] = test_augmented["proba"] * test_augmented["sample_weights"]
    probas = (test_augmented.groupby(["id"])["weighted_proba"].sum() / test_augmented.groupby(["id"])["sample_weights"].sum()).reset_index()
    probas.columns = ["id", "proba"]
    probas["label"] = (probas["proba"] > 0.5).astype(int)
    submission_format = datadict["submission_format"]
    assert set(submission_format.id) == set(probas.id)
    sf = submission_format.merge(probas.rename(columns={"proba": "p", "label": "l"}), how="left", on="id")
    sf["proba"] = sf["p"]
    sf["label"] = sf["l"]
    sf = sf[["id", "proba", "label"]]
    return sf, model



