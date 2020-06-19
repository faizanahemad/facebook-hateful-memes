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

from ..utils import in_notebook, get_device
from ..preprocessing import my_collate, make_weights_for_balanced_classes, TextImageDataset
import gc
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset
from transformers import optimization


def get_multistep_lr(milestones, gamma=0.2):
    def scheduler_init_fn(optimizer, epochs, batch_size, n_samples):
        n_steps = int(np.ceil(n_samples/batch_size))
        sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
        update_in_batch, update_in_epoch = False, True
        return sch, update_in_batch, update_in_epoch
    return scheduler_init_fn


def get_cosine_schedule_with_warmup(warmup_proportion=0.3):
    def init_fn(optimizer, epochs, batch_size, n_samples):
        n_steps = int(np.ceil(n_samples / batch_size))
        num_training_steps = n_steps * epochs
        num_warmup_steps = int(warmup_proportion * num_training_steps)
        sch = optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        update_in_batch, update_in_epoch = True, False
        return sch, update_in_batch, update_in_epoch
    return init_fn


def get_cosine_with_hard_restarts_schedule_with_warmup(warmup_proportion=0.2, num_cycles=1.0,):
    def init_fn(optimizer, epochs, batch_size, n_samples):
        n_steps = int(np.ceil(n_samples / batch_size))
        num_training_steps = n_steps * epochs
        num_warmup_steps = int(warmup_proportion * num_training_steps)
        sch = optimization.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles)
        update_in_batch, update_in_epoch = True, False
        return sch, update_in_batch, update_in_epoch
    return init_fn


def train(model, optimizer, scheduler_init_fn, batch_size, epochs, dataset, validation_strategy=None,
          plot=False,
          class_weights={0: 1, 1: 1.8}):
    if in_notebook():
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm as tqdm, trange

    if isinstance(dataset, TextImageDataset):
        use_images = dataset.use_images
        dataset.use_images = False
        training_fold_labels = torch.tensor(dataset.labels)
        dataset.use_images = use_images
    else:
        raise NotImplementedError()

    _ = model.train()

    weights = make_weights_for_balanced_classes(training_fold_labels, class_weights) # {0: 1, 1: 1.81} -> 0.814	0.705 || {0: 1, 1: 1.5}->0.796	0.702
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate,
                              shuffle=False, num_workers=32, pin_memory=True, sampler=sampler)
    train_losses = []
    learning_rates = []
    scheduler, update_in_batch, update_in_epoch = scheduler_init_fn(optimizer, epochs, batch_size, len(training_fold_labels)) if scheduler_init_fn is not None else (None, False, False)

    with trange(epochs) as epo:
        for epoc in epo:
            _ = model.train()
            if update_in_epoch:
                scheduler.step()
            _ = gc.collect()
            train_losses_cur_epoch = []
            with tqdm(train_loader) as data_batch:
                for batch in data_batch:
                    optimizer.zero_grad()
                    _, _, _, loss = model(batch)
                    loss.backward()
                    optimizer.step()
                    if update_in_batch:
                        scheduler.step()
                    train_losses.append(loss.cpu().item())
                    train_losses_cur_epoch.append(loss.cpu().item())
                    learning_rates.append(optimizer.param_groups[0]['lr'])
            print("Epoch = ", epoc + 1, "Loss = %.6f" % np.mean(train_losses_cur_epoch), "LR = %.8f" % optimizer.param_groups[0]['lr'])
            if validation_strategy is not None:
                if (epoc + 1) in validation_strategy["validation_epochs"]:
                    if "train" in validation_strategy:
                        vst, _ = validation_strategy["train"]["method"](*validation_strategy["train"]["args"])
                    if "val" in validation_strategy:
                        vsv, _ = validation_strategy["val"]["method"](*validation_strategy["val"]["args"])
                    vst = vst[-1]
                    vsv = vsv[-1]
                    print("Epoch = ", epoc + 1, "Train = %.6f" % vst, "Val = %.6f" % vsv,)

    import matplotlib.pyplot as plt
    if plot:
        print(model)
        t = list(range(len(train_losses)))

        fig, ax1 = plt.subplots(figsize=(8, 8))

        color = 'tab:red'
        ax1.set_xlabel('Training Batches')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(t, train_losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Learning Rate', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, learning_rates, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
    return train_losses, learning_rates


def generate_predictions(model, batch_size, dataset):
    _ = model.eval()
    proba_list = []
    predictions_list = []
    labels_list = []

    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate,
                             shuffle=False, num_workers=32, pin_memory=True)
    with torch.no_grad():
        for batch in test_loader:
            logits, _, _, _ = model(batch)
            labels = batch.label
            labels_list.extend(labels)
            logits = logits.cpu()
            top_class = logits.max(dim=1).indices
            top_class = top_class.flatten().tolist()
            probas = logits[:, 1].tolist()
            predictions_list.extend(top_class)
            proba_list.extend(probas)
    return proba_list, predictions_list, labels_list


def validate(model, batch_size, dataset, test_df):
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    proba_list, predictions_list, labels_list = generate_predictions(model, batch_size, dataset)

    test_df["proba"] = proba_list
    test_df["weighted_proba"] = test_df["proba"] * test_df["sample_weights"]
    probas = (test_df.groupby(["id"])["weighted_proba"].sum() / test_df.groupby(["id"])[
        "sample_weights"].sum()).reset_index()
    probas.columns = ["id", "proba"]
    probas["predictions_list"] = (probas["proba"] > 0.5).astype(int)

    proba_list = probas["proba"].values
    predictions_list = probas["predictions_list"].values
    labels_list = probas.merge(test_df[["id", "label"]], on="id")["label"].values
    auc = roc_auc_score(labels_list, proba_list)
    # p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(labels_list, predictions_list, average="micro")
    prfs = precision_recall_fscore_support(labels_list, predictions_list, average=None, labels=[0, 1])
    map = average_precision_score(labels_list, proba_list)
    acc = accuracy_score(labels_list, predictions_list)
    validation_scores = [map, acc, auc]
    return validation_scores, prfs


def model_builder(model_class, model_params,
                  optimiser_class=torch.optim.Adam, optimiser_params=dict(lr=0.001, weight_decay=1e-5)):
    def builder(**kwargs):
        prams = dict(model_params)
        prams.update(kwargs)
        model = model_class(**prams)
        model.to(get_device())
        optimizer = optimiser_class(filter(lambda p: p.requires_grad, model.parameters()), **optimiser_params)
        return model, optimizer

    return builder


def convert_dataframe_to_dataset(df, metadata, train=True):
    from functools import partial
    text = list(df.text)
    labels = df["label"].values if "label" in df else None
    sample_weights = df["sample_weights"].values if "sample_weights" in df else None
    ds = TextImageDataset(text, list(df.img), labels, sample_weights,
                          text_transform=metadata["train_text_transform"] if train else metadata["test_text_transform"],
                          image_transform=metadata["train_image_transform"] if train else metadata["test_image_transform"],
                          cache_images=metadata["cache_images"], use_images=metadata["use_images"])
    return ds


def random_split_for_augmented_dataset(datadict, augmentation_weights: Dict[str, float], n_splits=5, multi_eval=False, random_state=0):
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    metadata = datadict["metadata"]
    augmented_data = metadata["augmented_data"]
    train = datadict["train"]
    train["augmented"] = False
    train["augment_type"] = "None"
    train_augmented = datadict["train_augmented"] if augmented_data else train
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_idx, test_idx in skf.split(train, train.label):
        train_split = train.iloc[train_idx]
        test_split = train.iloc[test_idx]
        train_split = train_split[["id"]].merge(train_augmented, how="inner", on="id")
        test_split = test_split[["id"]].merge(train_augmented, how="inner", on="id")

        if not multi_eval:
            test_split = test_split[~test_split["augmented"]]
            test_split["sample_weights"] = 1.0
        else:
            test_split["sample_weights"] = 0.0
            for k, v in augmentation_weights.items():
                test_split.loc[test_split["augment_type"] == k, "sample_weights"] = v
            test_split = test_split[test_split["sample_weights"] > 0]
        train_split["sample_weights"] = 0.0
        for k, v in augmentation_weights.items():
            train_split.loc[train_split["augment_type"] == k, "sample_weights"] = v
        train_split = train_split[train_split["sample_weights"] > 0]
        yield (convert_dataframe_to_dataset(train_split, metadata, True),
               convert_dataframe_to_dataset(train_split, metadata, False),
               convert_dataframe_to_dataset(test_split, metadata, False), train_split, test_split)


def train_validate_ntimes(model_fn, data, batch_size, epochs,
                          augmentation_weights: Dict[str, float],
                          multi_eval=False, kfold=False, scheduler_init_fn=None,
                          random_state=None, validation_epochs=None):
    from tqdm import tqdm
    getattr(tqdm, '_instances', {}).clear()
    if in_notebook():
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm as tqdm, trange
    results_list = []
    prfs_list = []
    index = ["map", "accuracy", "auc"]
    model_stats_shown = False

    for training_fold_dataset, training_test_dataset, testing_fold_dataset, train_df, test_df in random_split_for_augmented_dataset(data, augmentation_weights,
                                                                                                                                    multi_eval=multi_eval,
                                                                                                                                    random_state=random_state):
        model, optimizer = model_fn(dataset=training_fold_dataset)
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        if not model_stats_shown:
            print("Model Params = %s" % (params), "\n", model)
            model_stats_shown = True
        validation_strategy = dict(validation_epochs=validation_epochs,
                                   train=dict(method=validate, args=[model, batch_size, training_test_dataset, train_df]),
                                   val=dict(method=validate, args=[model, batch_size, testing_fold_dataset, test_df]))
        validation_strategy = validation_strategy if validation_epochs is not None else None
        train_losses, learning_rates = train(model, optimizer, scheduler_init_fn, batch_size, epochs, training_fold_dataset,
                                             validation_strategy)

        validation_scores, prfs_val = validate(model, batch_size, testing_fold_dataset, test_df)
        train_scores, prfs_train = validate(model, batch_size, training_test_dataset, train_df)
        prfs_list.append(prfs_train + prfs_val)
        rdf = dict(train=train_scores, val=validation_scores)
        rdf = pd.DataFrame(data=rdf, index=index)
        results_list.append(rdf)
        if not kfold:
            break

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
    return results, prfs
