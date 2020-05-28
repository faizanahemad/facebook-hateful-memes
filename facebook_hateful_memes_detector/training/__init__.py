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
        pass
    elif isinstance(dataset, Subset):
        pass
    else:
        raise NotImplementedError()

    _ = model.train()
    training_fold_labels = torch.tensor([dataset[i][2] for i in range(len(dataset))])
    weights = make_weights_for_balanced_classes(training_fold_labels)
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
                    for texts, images, labels in data_batch:
                        optimizer.zero_grad()
                        _, _, _, _, loss = model(texts, images, labels)
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


def validate(model, batch_size, dataset):
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    proba_list, predictions_list, labels_list = generate_predictions(model, batch_size, dataset)
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
        for texts, images, labels in test_loader:
            logits, top_class, _, _, _ = model(texts, images, labels)
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


def train_validate_ntimes(model_fn, data, n_tests, batch_size, epochs):
    if in_notebook():
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm as tqdm, trange
    results_list = []
    prfs_list = []
    index = ["f1_micro", "map", "accuracy", "auc"]
    with trange(n_tests) as nt:
        for _ in nt:
            dataset = data["train"]
            size = len(dataset)
            training_fold_dataset, testing_fold_dataset = torch.utils.data.random_split(dataset, [int(size * 0.8),
                                                                                                  size - int(size * 0.8)])
            model, optimizer, scheduler = model_fn()
            train_losses, learning_rates = train(model, optimizer, scheduler, batch_size, epochs, training_fold_dataset)

            validation_scores, prfs_val = validate(model, batch_size, testing_fold_dataset)
            train_scores, prfs_train = validate(model, batch_size, training_fold_dataset)
            prfs_list.append(prfs_train + prfs_val)
            rdf = dict(train=train_scores, val=validation_scores)
            rdf = pd.DataFrame(data=rdf, index=index)
            results_list.append(rdf)
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


def train_and_predict(model_fn, data, batch_size, epochs):
    dataset = data["train"]
    model, optimizer, scheduler = model_fn()
    train_losses, learning_rates = train(model, optimizer, scheduler, batch_size, epochs, dataset, plot=True)
    test_dataset = data["test"]
    proba_list, predictions_list, labels_list = generate_predictions(model, batch_size, test_dataset)
    test = data["test_df"]
    submission = pd.DataFrame(dict(id=test.id, proba=proba_list, label=predictions_list),
                              columns=["id", "proba", "label"])
    return submission
