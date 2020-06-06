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

from ..utils import in_notebook
from ..preprocessing import my_collate, make_weights_for_balanced_classes, TextImageDataset
import gc
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset


def train(model, optimizer, scheduler, batch_size, epochs, dataset, plot=False):
    if in_notebook():
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm as tqdm, trange

    if isinstance(dataset, TextImageDataset):
        use_images = dataset.use_images
        dataset.use_images = False
        training_fold_labels = torch.tensor([dataset[i][2] for i in range(len(dataset))])
        dataset.use_images = use_images
    elif isinstance(dataset, Subset):
        use_images = dataset.dataset.use_images
        dataset.dataset.use_images = False
        training_fold_labels = torch.tensor([dataset[i][2] for i in range(len(dataset))])
        dataset.dataset.use_images = use_images
    else:
        raise NotImplementedError()

    _ = model.train()

    weights = make_weights_for_balanced_classes(training_fold_labels, {0: 1, 1: 1.8}) # {0: 1, 1: 1.81} -> 0.814	0.705 || {0: 1, 1: 1.5}->0.796	0.702
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate,
                              shuffle=False, num_workers=32, pin_memory=True, sampler=sampler)
    train_losses = []
    learning_rates = []
    try:
        with trange(epochs) as epo:
            for _ in epo:
                _ = gc.collect()
                with tqdm(train_loader) as data_batch:
                    for texts, images, labels, sample_weights in data_batch:
                        optimizer.zero_grad()
                        _, _, _, _, loss = model(texts, images, labels, sample_weights)
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        train_loss = loss.item() * labels.size(0)
                        train_losses.append(loss.item())
                        learning_rates.append(optimizer.param_groups[0]['lr'])
                    data_batch.clear()
                    data_batch.close()
        epo.clear()
        epo.close()

    except (KeyboardInterrupt, Exception) as e:
        epo.close()
        data_batch.close()
        raise
    epo.close()
    data_batch.close()
    import matplotlib.pyplot as plt

    if plot:
        print(model)
        t = list(range(len(train_losses)))

        fig, ax1 = plt.subplots()

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
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(labels_list, predictions_list, average="micro")
    prfs = precision_recall_fscore_support(labels_list, predictions_list, average=None, labels=[0, 1])
    map = average_precision_score(labels_list, proba_list)
    acc = accuracy_score(labels_list, predictions_list)
    validation_scores = [f1_micro, map, acc, auc]
    return validation_scores, prfs


def generate_predictions(model, batch_size, dataset):
    _ = model.eval()
    proba_list = []
    predictions_list = []
    labels_list = []

    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate,
                             shuffle=False, num_workers=32, pin_memory=True)
    with torch.no_grad():
        for texts, images, labels, sample_weights in test_loader:
            logits, top_class, _, _, _ = model(texts, images, labels, sample_weights)
            labels = labels.tolist()
            labels_list.extend(labels)
            top_class = top_class.flatten().tolist()
            probas = logits[:, 1].tolist()
            predictions_list.extend(top_class)
            proba_list.extend(probas)
    return proba_list, predictions_list, labels_list


def model_builder(model_class, model_params,
                  optimiser_class=torch.optim.Adam, optimiser_params=dict(lr=0.001, weight_decay=1e-5),
                  scheduler_class=None, scheduler_params=None):
    def builder(**kwargs):
        prams = dict(model_params)
        prams.update(kwargs)
        model = model_class(**prams)
        optimizer = optimiser_class(model.parameters(), **optimiser_params)
        scheduler = None
        if scheduler_class is not None:
            scheduler = scheduler_class(optimizer, **scheduler_params)
        return model, optimizer, scheduler

    return builder


def convert_dataframe_to_dataset(df, metadata, train=True):
    from functools import partial
    import os
    joiner = partial(os.path.join, metadata["data_dir"])
    text = list(df.text)
    img = list(map(joiner, df.img))
    labels = torch.tensor(df.label) if "label" in df else None
    sample_weights = torch.tensor(df.sample_weights) if "sample_weights" in df else None
    ds = TextImageDataset(text, img, labels, sample_weights,
                          text_transform=metadata["train_text_transform"] if train else metadata["test_text_transform"],
                          image_transform=metadata["train_image_transform"] if train else metadata["test_image_transform"],
                          cache_images=metadata["cache_images"], use_images=metadata["use_images"])
    return ds


def random_split_for_augmented_dataset(datadict, augmentation_weights: Dict[str, float], n_splits=5, multi_eval=False):
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    metadata = datadict["metadata"]
    augmented_data = metadata["augmented_data"]
    train = datadict["train"]
    train["augmented"] = False
    train["augment_type"] = "None"
    train_augmented = datadict["train_augmented"] if augmented_data else train
    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
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


def train_validate_ntimes(model_fn, data, n_tests, batch_size, epochs,
                          augmentation_weights: Dict[str, float],
                          multi_eval=False, kfold=False):
    from tqdm import tqdm
    getattr(tqdm, '_instances', {}).clear()
    if in_notebook():
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm as tqdm, trange
    results_list = []
    prfs_list = []
    index = ["f1_micro", "map", "accuracy", "auc"]
    model_stats_shown = False
    with trange(n_tests) as nt:
        for _ in nt:
            for training_fold_dataset, training_test_dataset, testing_fold_dataset, train_df, test_df in random_split_for_augmented_dataset(data, augmentation_weights, multi_eval=multi_eval):
                model, optimizer, scheduler = model_fn(dataset=training_fold_dataset)
                model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
                params = sum([np.prod(p.size()) for p in model_parameters])
                if not model_stats_shown:
                    print("Model Params = %s" % (params), "\n", model)
                train_losses, learning_rates = train(model, optimizer, scheduler, batch_size, epochs, training_fold_dataset)

                validation_scores, prfs_val = validate(model, batch_size, testing_fold_dataset, test_df)
                train_scores, prfs_train = validate(model, batch_size, training_test_dataset, train_df)
                prfs_list.append(prfs_train + prfs_val)
                rdf = dict(train=train_scores, val=validation_scores)
                rdf = pd.DataFrame(data=rdf, index=index)
                results_list.append(rdf)
                if not kfold:
                    break
    nt.close()
    results = np.stack(results_list, axis=0)
    means = pd.DataFrame(results.mean(0), index=index)
    stds = pd.DataFrame(results.std(0), index=index)
    tuples = [('mean', 'f1_micro'),
              ('mean', 'map'),
              ('mean', 'accuracy'),
              ('mean', 'auc'),
              ('std', 'f1_micro'),
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


def train_and_predict(model_fn, datadict, batch_size, epochs, augmentation_weights: Dict[str, float], multi_eval=False):
    train = datadict["train"]
    train["augmented"] = False
    train["augment_type"] = "None"
    metadata = datadict["metadata"]
    augmented_data = metadata["augmented_data"]
    train_augmented = datadict["train_augmented"] if augmented_data else train
    train_augmented["sample_weights"] = 0.0
    for k, v in augmentation_weights.items():
        train_augmented.loc[train_augmented["augment_type"] == k, "sample_weights"] = v
    train_augmented = train_augmented[train_augmented["sample_weights"] > 0]
    dataset = convert_dataframe_to_dataset(train_augmented, metadata, True)

    model, optimizer, scheduler = model_fn(dataset=dataset)
    train_losses, learning_rates = train(model, optimizer, scheduler, batch_size, epochs, dataset, plot=True)
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
    return probas
