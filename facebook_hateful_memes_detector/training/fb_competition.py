from typing import List, Dict, Union, Callable, Tuple

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import contractions
import pandas as pd

from ..utils import in_notebook, CNNHead, AveragedLinearHead, OneTokenPositionLinearHead, MultiTaskForward, CNN2DHead, DecoderEnsemblingHead
from ..preprocessing import my_collate, make_weights_for_balanced_classes, TextImageDataset
import gc
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset
from transformers import optimization
from .generic import *


def fb_1d_loss_builder(n_dims, n_tokens, n_out, dropout, **kwargs):
    loss = kwargs.pop("loss", "classification")
    classification_head = kwargs.pop("classification_head", "cnn1d")
    if classification_head == "cnn1d":
        head = CNNHead(n_dims, n_tokens, n_out, dropout, loss, **kwargs)
    elif classification_head == "decoder_ensemble":
        head = DecoderEnsemblingHead(n_dims, n_tokens, n_out, dropout, loss, **kwargs)
    else:
        raise NotImplementedError
    mtf = MultiTaskForward([head])
    return mtf


def train_and_predict(model_fn: Union[Callable, Tuple], datadict, batch_size, epochs,
                      accumulation_steps=1, scheduler_init_fn=None,
                      model_call_back=None, validation_epochs=None,
                      sampling_policy=None, class_weights=None,
                      ):
    train_df = datadict["train"]
    dev_df = datadict["dev"]
    metadata = datadict["metadata"]
    dataset = convert_dataframe_to_dataset(train_df, metadata, True)
    dev_dataset = convert_dataframe_to_dataset(dev_df, metadata, True)
    if callable(model_fn):
        model, optimizer = model_fn()
    else:
        model, optimizer = model_fn
    validation_strategy = dict(validation_epochs=validation_epochs,
                               train=dict(method=validate, args=[model, batch_size, dataset], kwargs=dict(display_detail=False)),
                               val=dict(method=validate, args=[model, batch_size, dev_dataset], kwargs=dict(display_detail=True)))
    validation_strategy = validation_strategy if validation_epochs is not None else None
    train_losses, learning_rates = train(model, optimizer, scheduler_init_fn, batch_size, epochs, dataset,
                                         model_call_back=model_call_back, validation_strategy=validation_strategy,
                                         accumulation_steps=accumulation_steps, plot=True,
                                         sampling_policy=sampling_policy, class_weights=class_weights)
    return predict(model, datadict, batch_size)


def predict(model, datadict, batch_size):
    metadata = datadict["metadata"]
    test = datadict["test"]
    ids = test["id"] if "id" in test.columns else test["ID"]
    id_name = "id" if "id" in test.columns else "ID"
    test_dataset = convert_dataframe_to_dataset(test, metadata, False)
    proba_list, all_probas_list, predictions_list, labels_list = generate_predictions(model, batch_size, test_dataset, collate_fn=my_collate)
    probas = pd.DataFrame({id_name: ids, "proba": proba_list, "label": predictions_list})
    sf = probas
    if "submission_format" in datadict and type(datadict["submission_format"]) == pd.DataFrame and len(datadict["submission_format"]) == len(probas):
        submission_format = datadict["submission_format"]
        assert set(submission_format.id) == set(probas.id)
        sf = submission_format.merge(probas.rename(columns={"proba": "p", "label": "l"}), how="inner", on="id")
        sf["proba"] = sf["p"]
        sf["label"] = sf["l"]
        sf = sf[["id", "proba", "label"]]
    return sf, model



