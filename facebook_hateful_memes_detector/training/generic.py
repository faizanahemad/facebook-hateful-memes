from typing import List, Dict

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

from ..utils import in_notebook, get_device, dict2sampleList, clean_memory, GaussianNoise
from ..preprocessing import my_collate, make_weights_for_balanced_classes, TextImageDataset
import gc
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset
from transformers import optimization
from .model_params import group_wise_lr, group_wise_finetune
from collections import Counter


def calculate_auc_dice_loss(logits, labels, loss, auc_loss_coef, dice_loss_coef, loss_coef, ):
    binary = logits.size(1) == 2
    eps = 1e-5
    if binary:
        auc_loss = 0.0
        if auc_loss_coef > 0:
            probas = logits[:, 1]
            pos_probas = labels * probas
            neg_probas = (1 - labels) * probas
            neg_probas = neg_probas[neg_probas > eps]
            pos_probas = pos_probas[pos_probas > eps]

            pos_probas = pos_probas[pos_probas < (1 - eps)]
            neg_probas = neg_probas[neg_probas < (1 - eps)]  # TODO: Is this required

            num_entries = max(int(len(labels) / 8), 1)
            loss_1, loss_2 = 0.0, 0.0
            num_entries_neg, num_entries_pos = 0, 0

            if len(neg_probas) > 1:
                pos_probas = pos_probas[pos_probas < neg_probas.max()]
                num_entries_neg = min(num_entries, int(len(neg_probas) / 2))

            if len(pos_probas) > 1:
                neg_probas = neg_probas[neg_probas > pos_probas.min()]
                num_entries_pos = min(num_entries, int(len(pos_probas) / 2))

            if 1 <= num_entries_neg < len(neg_probas):
                neg_probas_max = torch.topk(neg_probas, num_entries_neg, 0).values.mean()
                loss_2 = (neg_probas_max - pos_probas).mean()

            if 1 <= num_entries_pos < len(pos_probas):
                pos_probas_min = torch.topk(pos_probas, num_entries_pos, 0, largest=False).values.mean()
                loss_1 = (neg_probas - pos_probas_min).mean()
            auc_loss = loss_1 + loss_2

        dice_loss = torch.prod(logits, 1).mean()
        loss = (loss_coef * loss + auc_loss_coef * auc_loss + dice_loss_coef * dice_loss) / (loss_coef + auc_loss_coef + dice_loss_coef)

    return loss



def get_regularizer_scheduler(warmup_proportion=0.7):
    def scheduler(model, batch, num_batches, epoch, num_epochs):
        total_batches = num_batches * num_epochs
        cur_batch = num_batches * epoch + batch
        warmup_batches = warmup_proportion * total_batches
        for layer, param in model.reg_layers:
            new_param = np.interp(cur_batch,
                                  [0, warmup_batches, total_batches],
                                  [param, param, 0])
            if layer.__class__ == GaussianNoise:
                layer.sigma = new_param
            elif layer.__class__ == nn.Dropout:
                layer.p = new_param
            else:
                raise NotImplementedError

    return scheduler



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


def train(model, optimizer, scheduler_init_fn,
          batch_size, epochs, dataset,
          model_call_back=None, accumulation_steps=1,
          validation_strategy=None,
          plot=False,
          sampling_policy=None,
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

    assert accumulation_steps >= 1 and type(accumulation_steps) == int
    _ = model.train()
    use_autocast = False
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        use_autocast = "cuda" in str(get_device())
    except:
        pass

    assert sampling_policy is None or sampling_policy in ["with_replacement", "without_replacement", "without_replacement_v2", "without_replacement_v3"]
    if sampling_policy == "with_replacement":
        weights = make_weights_for_balanced_classes(training_fold_labels, class_weights)  # {0: 1, 1: 1.81} -> 0.814	0.705 || {0: 1, 1: 1.5}->0.796	0.702
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        examples = len(training_fold_labels)
        divisor = 1
    elif sampling_policy == "without_replacement":
        # num lowest class * num classes
        weights = make_weights_for_balanced_classes(training_fold_labels, class_weights)  # {0: 1, 1: 1.81} -> 0.814	0.705 || {0: 1, 1: 1.5}->0.796	0.702
        sampler = WeightedRandomSampler(weights, int(len(weights)/2), replacement=False)
        divisor = 2
        examples = int(len(weights)/2)
        shuffle = False
    elif sampling_policy == "without_replacement_v2":
        cnt = Counter(training_fold_labels.tolist())
        examples = cnt.most_common()[-1][1] * len(cnt)
        weights = make_weights_for_balanced_classes(training_fold_labels, class_weights)  # {0: 1, 1: 1.81} -> 0.814	0.705 || {0: 1, 1: 1.5}->0.796	0.702
        sampler = WeightedRandomSampler(weights, examples, replacement=False)
        divisor = float(len(training_fold_labels)) / examples
        shuffle = False
    elif sampling_policy == "without_replacement_v3":
        cnt = Counter(training_fold_labels.tolist())
        examples = int(cnt.most_common()[-1][1] * len(cnt) / 2)
        weights = make_weights_for_balanced_classes(training_fold_labels, class_weights)  # {0: 1, 1: 1.81} -> 0.814	0.705 || {0: 1, 1: 1.5}->0.796	0.702
        sampler = WeightedRandomSampler(weights, examples, replacement=False)
        divisor = float(len(training_fold_labels)) / examples
        shuffle = False
    else:
        sampler = None
        shuffle = True
        examples = len(training_fold_labels)
        divisor = 1
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate,
                              shuffle=shuffle, num_workers=get_global("dataloader_workers"), pin_memory=True, sampler=sampler)

    train_losses = []
    learning_rates = []
    epochs = int(epochs * divisor)
    scheduler, update_in_batch, update_in_epoch = scheduler_init_fn(optimizer, epochs, batch_size, examples) if scheduler_init_fn is not None else (None, False, False)
    print("Autocast = ", use_autocast, "Epochs = ", epochs, "Divisor =", divisor, "Examples =", examples, "Batch Size = ", batch_size,)
    print("Training Samples = ", len(training_fold_labels), "Weighted Sampling = ", sampler is not None,
          "Num Batches = ", len(train_loader), "Accumulation steps = ", accumulation_steps)
    if len(train_loader) % accumulation_steps != 0:
        print("[WARN]: Number of training batches not divisible by accumulation steps, some training batches will be wasted due to this.")
    with trange(epochs) as epo:
        # TODO Reduce regularization of model in last few epochs, this way model is acquainted to work with real less regularized data (Real data distribution).
        for epoc in epo:
            _ = model.train()
            optimizer.zero_grad()
            if update_in_epoch:
                scheduler.step()
            clean_memory()
            train_losses_cur_epoch = []
            with tqdm(train_loader) as data_batch:
                for batch_idx, batch in enumerate(data_batch):
                    if model_call_back is not None:
                        model_call_back(model, batch_idx, len(train_loader), epoc, epochs)
                    if use_autocast:
                        with autocast():
                            _, _, _, loss = model(batch)
                            loss = loss / accumulation_steps

                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        _, _, _, loss = model(batch)
                        loss = loss / accumulation_steps
                        loss.backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    if update_in_batch:
                        scheduler.step()
                    train_losses.append(float(loss.cpu().detach().item()))
                    train_losses_cur_epoch.append(float(loss.cpu().detach().item()))
                    learning_rates.append(float(optimizer.param_groups[0]['lr']))
                    clean_memory()
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

    if plot:
        plot_loss_lr(train_losses, learning_rates)
    return train_losses, learning_rates


def train_for_augment_similarity(model, optimizer, scheduler_init_fn,
                                 batch_size, epochs, dataset,
                                 augment_method,
                                 model_call_back=None, accumulation_steps=1,
                                 plot=False):
    import copy
    orig_model = copy.deepcopy(model)
    from tqdm.auto import tqdm as tqdm, trange


    assert accumulation_steps >= 1 and type(accumulation_steps) == int
    _ = model.train()
    use_autocast = False
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        use_autocast = "cuda" in str(get_device())
    except:
        pass

    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate,
                              shuffle=True, num_workers=get_global("dataloader_workers"), pin_memory=True, sampler=None)

    examples = len(dataset)
    train_losses = []
    learning_rates = []
    scheduler, update_in_batch, update_in_epoch = scheduler_init_fn(optimizer, epochs, batch_size, examples) if scheduler_init_fn is not None else (None, False, False)
    print("Autocast = ", use_autocast, "Epochs = ", epochs, "Examples =", examples, "Batch Size = ", batch_size,)
    print("Training Samples = ", len(dataset), "Weighted Sampling = ", False,
          "Num Batches = ", len(train_loader), "Accumulation steps = ", accumulation_steps)

    if len(train_loader) % accumulation_steps != 0:
        print("[WARN]: Number of training batches not divisible by accumulation steps, some training batches will be wasted due to this.")
    with trange(epochs) as epo:
        # TODO Reduce regularization of model in last few epochs, this way model is acquainted to work with real less regularized data (Real data distribution).
        for epoc in epo:
            _ = model.train()
            optimizer.zero_grad()
            if update_in_epoch:
                scheduler.step()
            clean_memory()
            train_losses_cur_epoch = []
            with tqdm(train_loader) as data_batch:
                for batch_idx, batch in enumerate(data_batch):
                    if model_call_back is not None:
                        model_call_back(model, batch_idx, len(train_loader), epoc, epochs)
                    if use_autocast:
                        with autocast():
                            repr = model(augment_method(batch))
                            orig_repr = orig_model(batch)
                            loss = ((repr - orig_repr)**2).mean()
                            loss = loss / accumulation_steps

                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        repr = model(augment_method(batch))
                        orig_repr = orig_model(batch)
                        loss = ((repr - orig_repr) ** 2).mean()
                        loss = loss / accumulation_steps
                        loss.backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    if update_in_batch:
                        scheduler.step()
                    train_losses.append(float(loss.cpu().detach().item()))
                    train_losses_cur_epoch.append(float(loss.cpu().detach().item()))
                    learning_rates.append(float(optimizer.param_groups[0]['lr']))
                    clean_memory()
            print("Epoch = ", epoc + 1, "Loss = %.6f" % np.mean(train_losses_cur_epoch), "LR = %.8f" % optimizer.param_groups[0]['lr'])

    if plot:
        plot_loss_lr(train_losses, learning_rates)
    return train_losses, learning_rates


def plot_loss_lr(train_losses, learning_rates):
    import matplotlib.pyplot as plt
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


def generate_predictions(model, batch_size, dataset):
    _ = model.eval()
    proba_list = []
    predictions_list = []
    labels_list = []
    clean_memory()
    batch_size = batch_size + int(batch_size/2)
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate,
                             shuffle=False, num_workers=get_global("dataloader_workers"), pin_memory=True)

    use_autocast = False
    try:
        from torch.cuda.amp import autocast
        use_autocast = "cuda" in str(get_device())
    except:
        pass
    with torch.no_grad():
        clean_memory()
        for batch in test_loader:
            if use_autocast:
                with autocast():
                    logits, _, _, _ = model(batch)
            else:
                logits, _, _, _ = model(batch)
            labels = batch["label"]
            labels_list.extend(labels)
            logits = logits.cpu().detach()
            top_class = logits.max(dim=1).indices
            top_class = top_class.flatten().tolist()
            probas = logits[:, 1].tolist()
            predictions_list.extend(top_class)
            proba_list.extend(probas)
            clean_memory()
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
    auc = roc_auc_score(labels_list, proba_list, multi_class="ovo", average="macro")
    # p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(labels_list, predictions_list, average="micro")
    prfs = precision_recall_fscore_support(labels_list, predictions_list, average=None, labels=[0, 1])
    map = average_precision_score(labels_list, proba_list)
    acc = accuracy_score(labels_list, predictions_list)
    validation_scores = [map, acc, auc]
    return validation_scores, prfs


def model_builder(model_class, model_params,
                  optimiser_class=torch.optim.AdamW, per_param_opts_fn=None,
                  optimiser_params=dict(lr=0.001, weight_decay=1e-5)):
    def builder(**kwargs):
        prams = dict(model_params)
        prams.update(kwargs)
        model = model_class(**prams)
        model.to(get_device())
        all_params = filter(lambda p: p.requires_grad, model.parameters())

        if per_param_opts_fn is not None:
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            if type(per_param_opts_fn) == list:
                params_conf = per_param_opts_fn
            elif type(per_param_opts_fn) == dict:
                params_conf, _ = group_wise_lr(model, per_param_opts_fn)
                _ = group_wise_finetune(model, per_param_opts_fn)
            else:
                raise NotImplementedError()
            assert type(params_conf) == list
            assert len(params_conf) > 0
            assert all(["params" in p for p in params_conf])

            all_params = params_conf

        optimizer = optimiser_class(all_params, **optimiser_params)
        return model, optimizer

    return builder


def convert_dataframe_to_dataset(df, metadata, train=True):
    from functools import partial
    text = list(df.text)
    labels = df["label"].values if "label" in df else None
    sample_weights = df["sample_weights"].values if "sample_weights" in df else None
    ds = TextImageDataset(text, list(df.img), labels, sample_weights,
                          text_transform=metadata["train_text_transform"] if train else metadata["test_text_transform"],
                          torchvision_image_transform=metadata["train_torchvision_image_transform"] if train else metadata["test_torchvision_image_transform"],
                          image_transform=metadata["train_image_transform"] if train else metadata["test_image_transform"],
                          cache_images=metadata["cache_images"], use_images=metadata["use_images"],
                          keep_original_text=metadata["keep_original_text"], keep_original_image=metadata["keep_original_image"],
                          keep_processed_image=metadata["keep_processed_image"], keep_torchvision_image=metadata["keep_torchvision_image"])
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
                          accumulation_steps=1,
                          multi_eval=False, kfold=False, scheduler_init_fn=None, model_call_back=None,
                          random_state=None, validation_epochs=None, show_model_stats=False,
                          sampling_policy=None,
                          class_weights={0: 1, 1: 1.8}):
    from tqdm import tqdm
    getattr(tqdm, '_instances', {}).clear()
    if in_notebook():
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm as tqdm, trange
    results_list = []
    prfs_list = []
    index = ["map", "accuracy", "auc"]

    for training_fold_dataset, training_test_dataset, testing_fold_dataset, train_df, test_df in random_split_for_augmented_dataset(data, augmentation_weights,
                                                                                                                                    multi_eval=multi_eval,
                                                                                                                                    random_state=random_state):
        model, optimizer = model_fn()
        if show_model_stats:
            model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            params = sum([np.prod(p.size()) for p in model_parameters])
            print("Trainable Params = %s" % (params), "\n", model)
            show_model_stats = not show_model_stats
        validation_strategy = dict(validation_epochs=validation_epochs,
                                   train=dict(method=validate, args=[model, batch_size, training_test_dataset, train_df]),
                                   val=dict(method=validate, args=[model, batch_size, testing_fold_dataset, test_df]))
        validation_strategy = validation_strategy if validation_epochs is not None else None
        train_losses, learning_rates = train(model, optimizer, scheduler_init_fn, batch_size, epochs, training_fold_dataset, model_call_back, accumulation_steps,
                                             validation_strategy, plot=not kfold, sampling_policy=sampling_policy, class_weights=class_weights)

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

