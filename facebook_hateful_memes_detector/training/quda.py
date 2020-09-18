import os
from typing import List, Dict, Callable, Tuple, Union

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..utils.globals import get_global
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import contractions
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..utils import in_notebook, get_device, dict2sampleList, clean_memory, GaussianNoise, my_collate, WordMasking
from ..preprocessing import make_weights_for_balanced_classes, TextImageDataset, make_weights_for_uda, make_sqrt_weights_for_balanced_classes, make_sqrt_weights_for_uda
import gc
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from torch.utils.data import Subset
from ..utils.sample import *
from transformers import optimization
from .model_params import group_wise_lr, group_wise_finetune
from collections import Counter
from IPython.display import display
from .generic import *
from .fb_competition import predict
from torch.utils.checkpoint import checkpoint


class LabelConsistencyDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, train: torch.utils.data.Dataset, test: torch.utils.data.Dataset,
                 num_classes: int,
                 augs: List[Callable]):
        self.train = train
        self.test = test
        self.augs = augs
        self.labels = list(self.train.labels) + [num_classes] * len(test)
        self.l1 = len(self.train)
        self.l2 = len(self.test)

    def __getitem__(self, item):
        label = self.labels[item]
        try:
            sample = self.train[item] if item < self.l1 else self.test[item - self.l1]
            sample.label = label
            return [aug(sample) for aug in self.augs]
        except Exception as e:
            print(item, len(self.train), self.__len__())
            raise e

    def __len__(self):
        return len(self.labels)


def label_consistency_collate(batch):
    samples = zip(*batch)
    return [SampleList(s) for s in samples]


class ModelWrapperForConsistency:
    def __init__(self, model, num_classes, consistency_loss_weight):
        self.model = model
        self.num_classes = num_classes
        self.consistency_loss_weight = consistency_loss_weight
        self.reg_layers = model.reg_layers if hasattr(model, "reg_layers") else []

    def __call__(self, batch):
        samples = batch
        if self.model.training:
            results = [checkpoint(self.model, s) for s in samples]
            rmse_loss = 0.0
            loss = 0.0
            logits = None
            res1 = results[0]
            for idx, res1 in enumerate(results):
                loss += res1[-1]
                logits = (logits + res1[0]) if logits is not None else res1[0]
                for res2 in results[idx+1:]:
                    rmse_loss = rmse_loss + self.consistency_loss_weight * self.num_classes * F.mse_loss(res1[0], res2[0])
            rmse_loss = rmse_loss / sum(list(range(len(results))))
            logits = logits / len(results)
            loss = loss / len(results)
            loss = loss + rmse_loss
        else:
            s1 = samples[0]
            res1 = self.model(s1)
            loss = res1[-1]
            logits = res1[0]
        return logits, res1[1], res1[2], loss

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)


def train_validate_ntimes(model_fn, data, batch_size, epochs,
                          accumulation_steps=1,
                          scheduler_init_fn=None, model_call_back=None,
                          random_state=None, validation_epochs=None, show_model_stats=False,
                          sampling_policy=None,
                          class_weights=None,
                          prediction_iters=1,
                          evaluate_in_train_mode=False, consistency_loss_weight=0.0, num_classes=2,
                          augs: List[Callable]=[identity],
                          ):
    from tqdm import tqdm
    getattr(tqdm, '_instances', {}).clear()
    from tqdm.auto import tqdm as tqdm, trange
    results_list = []
    prfs_list = []
    index = ["map", "accuracy", "auc"]
    metadata = data["metadata"]
    test_dev = data["metadata"]["test_dev"]
    actual_test = data["test"]

    trin = data["train"]
    dev = data["dev"]
    test = data["test"]
    metadata = data["metadata"]

    assert consistency_loss_weight >= 0
    training_fold_dataset = convert_dataframe_to_dataset(trin, metadata, consistency_loss_weight == 0,
                                                         numbers=data["numeric_train"], embed1=data["embed1_train"], embed2=data["embed2_train"])
    training_test_dataset = convert_dataframe_to_dataset(trin, metadata, False, cached_images=training_fold_dataset.images,
                                                         numbers=data["numeric_train"], embed1=data["embed1_train"], embed2=data["embed2_train"])
    testing_fold_dataset = convert_dataframe_to_dataset(dev, metadata, False, numbers=data["numeric_dev"], embed1=data["embed1_dev"], embed2=data["embed2_dev"])

    if callable(model_fn):
        model, optimizer = model_fn()
    else:
        model, optimizer = model_fn
    if show_model_stats:
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Trainable Params = %s" % (params), "\n", model)
    validation_strategy = dict(validation_epochs=validation_epochs,
                               train=dict(method=validate, args=[model, batch_size, training_test_dataset], kwargs=dict(display_detail=False)),
                               val=dict(method=validate, args=[model, batch_size, testing_fold_dataset], kwargs=dict(display_detail=False,
                                                                                                                     prediction_iters=prediction_iters,
                                                                                                                     evaluate_in_train_mode=evaluate_in_train_mode)))
    validation_strategy = validation_strategy if validation_epochs is not None else None
    if consistency_loss_weight > 0:
        tmodel = ModelWrapperForConsistency(model, num_classes, consistency_loss_weight)
        testing_dataset = convert_dataframe_to_dataset(actual_test, metadata, False)
        from torch.utils.data import ConcatDataset
        train_dataset = LabelConsistencyDatasetWrapper(training_fold_dataset, ConcatDataset((testing_fold_dataset, testing_dataset)), num_classes, augs)
        collate_fn = label_consistency_collate
    else:
        tmodel = model
        train_dataset = training_fold_dataset
        collate_fn = my_collate
    tmodel.to(get_device())
    train_losses, learning_rates, validation_stats = train(tmodel, optimizer, scheduler_init_fn, batch_size, epochs, train_dataset, model_call_back,
                                                           accumulation_steps,
                                                           validation_strategy, plot=True, sampling_policy=sampling_policy,
                                                           class_weights=class_weights, collate_fn=collate_fn, )

    validation_scores, prfs_val = validate(model, batch_size, testing_fold_dataset, display_detail=True)
    train_scores, prfs_train = validate(model, batch_size, training_test_dataset, display_detail=False, prediction_iters=prediction_iters,
                                                                                                        evaluate_in_train_mode=evaluate_in_train_mode)
    prfs_list.append(prfs_train + prfs_val)
    rdf = dict(train=train_scores, val=validation_scores)
    rdf = pd.DataFrame(data=rdf, index=index)
    results_list.append(rdf)

    results = np.stack(results_list, axis=0)
    means = pd.DataFrame(results.mean(0), index=index)
    stds = pd.DataFrame(results.std(0), index=index)
    tuples = [('mean', 'map'),
              ('mean', 'accuracy'),
              ('mean', 'auc'),
              ('std', 'map'),
              ('std', 'accuracy'),
              ('std', 'auc')]
    rowidx = pd.MultiIndex.from_tuples(tuples, names=['mean_or_std', 'metric'])
    results = pd.concat([means, stds], axis=0)
    results.index = rowidx
    results.columns = ["train", "val"]
    prfs_idx = pd.MultiIndex.from_product([["train", "val"], ["precision", "recall", "f1", "supoort"]])
    prfs = pd.DataFrame(np.array(prfs_list).mean(0), columns=["neg", "pos"], index=prfs_idx).T
    return results, prfs, validation_stats


def train_and_predict(model_fn: Union[Callable, Tuple], datadict, batch_size, epochs,
                      accumulation_steps=1, scheduler_init_fn=None,
                      model_call_back=None, validation_epochs=None,
                      sampling_policy=None, class_weights=None,
                      prediction_iters=1, evaluate_in_train_mode=False,
                      consistency_loss_weight=0.0, num_classes=2,
                      augs: List[Callable]=[identity],
                      show_model_stats=False, give_probas=True):
    train_df = datadict["train"]
    dev_df = datadict["dev"]
    test_df = datadict["test"]
    metadata = datadict["metadata"]
    dataset = convert_dataframe_to_dataset(train_df, metadata, consistency_loss_weight == 0,
                                           numbers=datadict["numeric_train"], embed1=datadict["embed1_train"], embed2=datadict["embed2_train"])
    dev_dataset = convert_dataframe_to_dataset(dev_df, metadata, False, numbers=datadict["numeric_dev"],
                                               embed1=datadict["embed1_dev"], embed2=datadict["embed2_dev"])
    test_dataset = convert_dataframe_to_dataset(test_df, metadata, False, numbers=datadict["numeric_test"],
                                                embed1=datadict["embed1_test"], embed2=datadict["embed2_test"])
    if callable(model_fn):
        model, optimizer = model_fn()
    else:
        model, optimizer = model_fn
    if show_model_stats:
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Trainable Params = %s" % (params), "\n", model)
        show_model_stats = not show_model_stats
    validation_strategy = dict(validation_epochs=validation_epochs,
                               train=dict(method=validate, args=[model, batch_size, dataset], kwargs=dict(display_detail=False)),
                               val=dict(method=validate, args=[model, batch_size, dev_dataset], kwargs=dict(display_detail=False,
                                                                                                            prediction_iters=prediction_iters,
                                                                                                            evaluate_in_train_mode=evaluate_in_train_mode)))
    validation_strategy = validation_strategy if validation_epochs is not None else None
    if consistency_loss_weight > 0:
        from torch.utils.data import ConcatDataset
        tmodel = ModelWrapperForConsistency(model, num_classes, consistency_loss_weight)
        train_dataset = LabelConsistencyDatasetWrapper(dataset, ConcatDataset((dev_dataset, test_dataset)), num_classes, augs)
        collate_fn = label_consistency_collate
    else:
        tmodel = model
        train_dataset = dataset
        collate_fn = my_collate
    tmodel.to(get_device())
    train_losses, learning_rates, validation_stats = train(tmodel, optimizer, scheduler_init_fn, batch_size, epochs, train_dataset,
                                                           model_call_back=model_call_back, validation_strategy=validation_strategy,
                                                           accumulation_steps=accumulation_steps, plot=True,
                                                           sampling_policy=sampling_policy, class_weights=class_weights, collate_fn=collate_fn)
    return predict(model, datadict, batch_size, prediction_iters=prediction_iters, evaluate_in_train_mode=evaluate_in_train_mode, give_probas=give_probas), model, validation_stats
