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

from ..utils import in_notebook, get_device, dict2sampleList, clean_memory, GaussianNoise, my_collate
from ..preprocessing import make_weights_for_balanced_classes, TextImageDataset
import gc
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Subset
from mmf.common.sample import Sample, SampleList
from transformers import optimization
from .model_params import group_wise_lr, group_wise_finetune
from collections import Counter
from IPython.display import display


def try_dice():
    loss = get_dice_loss(2, threshold_1=0.8, threshold_2=1e-2)
    import matplotlib.pyplot as plt
    x = torch.arange(0, 1, 0.01).detach().cpu()
    x2 = 1 - x
    xx = torch.cat([x.reshape(-1, 1), x2.reshape(-1, 1)], 1)
    y = loss(xx).detach().cpu()
    plt.plot(x, y)
    plt.show()


def try_dice_n_classes(n_classes):
    loss = get_dice_loss(n_classes, threshold_1=0.8, threshold_2=1e-2)
    import matplotlib.pyplot as plt
    x = torch.softmax(torch.randn(10000, n_classes), dim=1)
    y = loss(x).detach().cpu()
    x = torch.abs(x - 1/n_classes).mean(1)
    x, y = zip(*sorted(list(zip(x, y)), key=lambda k: k[0]))
    plt.plot(x, y)
    plt.show()


def mean_sigma_finder_for_dice_loss(n_classes, threshold_1=0.8, threshold_2=1e-2):
    """
    For given n_classes this provides a sigma value such that when any class proba goes above threshold_1, loss goes below threshold_2
    :param n_classes:
    :param threshold_1:
    :param threshold_2:
    :return:
    """
    x = torch.tensor([[threshold_1] + [(1 - threshold_1)/(n_classes - 1)] * (n_classes - 1)])
    candidates = np.logspace(np.log(0.5), np.log(1e-5), num=100000, base=np.e)
    mean = 1/n_classes
    for sigma in candidates:
        loss = dice_loss(x, mean=mean, sigma=sigma)
        if loss <= threshold_2:
            return mean, sigma

    raise ValueError


def dice_loss(logits, mean=0.5, sigma=0.125):
    x = torch.abs(logits - mean).mean(1)
    return normal_vector(x, sigma=sigma)


def normal_vector(x, mean=0.0, sigma=0.125):
    root_two_pi_sigma = sigma * 2.5066282
    exponent = torch.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2)))
    return exponent/root_two_pi_sigma


def get_dice_loss(n_classes, dice_loss_coef=1.0, threshold_1=0.8, threshold_2=1e-2):
    def zero_fn(logits, labels=None):
        return 0.0
    if dice_loss_coef == 0:
        return zero_fn
    mean, sigma = mean_sigma_finder_for_dice_loss(n_classes, threshold_1, threshold_2)

    def loss(logits, labels=None):
        dl = dice_loss(logits, mean, sigma).mean()
        return dice_loss_coef * dl
    return loss


def get_auc_loss(n_classes, auc_loss_coef, auc_method=6):
    def zeros_auc(logits, labels=None):
        return 0.0
    assert auc_loss_coef >= 0
    if n_classes > 2 or auc_loss_coef == 0:
        return zeros_auc

    def loss_method_1(logits, labels=None):
        if labels is None:
            return 0.0
        probas = logits[:, 1]
        pos_probas = labels * probas
        neg_probas = (1 - labels) * (1 - logits[:, 0])
        neg_proba_max = neg_probas.max().detach()
        pos_proba_min = pos_probas.min().detach()
        loss_1 = F.leaky_relu(neg_probas - pos_proba_min).mean()
        loss_2 = F.leaky_relu(neg_proba_max - pos_probas).mean()
        return auc_loss_coef * (loss_1 + loss_2)/2

    def loss_method_2(logits, labels=None):
        if labels is None:
            return 0.0
        probas = logits[:, 1]
        pos_probas = labels * probas
        neg_probas = (1 - labels) * (1 - logits[:, 0])
        neg_proba_max = neg_probas.max().detach()
        pos_proba_min = pos_probas.min().detach()
        loss_1, loss_2 = 0.0, 0.0

        pos_probas = pos_probas[pos_probas < neg_proba_max]
        neg_probas = neg_probas[neg_probas > pos_proba_min]

        num_entries_neg = max(1, int(len(neg_probas) / 4))
        num_entries_pos = max(1, int(len(pos_probas) / 4))
        if 1 <= num_entries_neg <= len(neg_probas):
            neg_probas_max = torch.topk(neg_probas, num_entries_neg, 0).values.mean()
            # neg_probas_max = neg_probas.max()
            loss_2 = (neg_probas_max - pos_probas).mean()

        if 1 <= num_entries_pos <= len(pos_probas):
            pos_probas_min = torch.topk(pos_probas, num_entries_pos, 0, largest=False).values.mean()
            # pos_probas_min = pos_probas.min()
            loss_1 = (neg_probas - pos_probas_min).mean()

        return auc_loss_coef * (loss_1 + loss_2) / 2

    def loss_method_3(logits, labels=None):
        if labels is None:
            return 0.0
        probas = logits[:, 1]
        pos_probas = labels * probas
        neg_probas = (1 - labels) * (1 - logits[:, 0])
        neg_proba_max = neg_probas.max().detach()
        pos_proba_min = pos_probas.min().detach()
        loss_1, loss_2 = 0.0, 0.0

        pos_probas = pos_probas[pos_probas < neg_proba_max]
        neg_probas = neg_probas[neg_probas > pos_proba_min]

        if 1 <= len(neg_probas):
            loss_2 = (neg_proba_max - pos_probas).mean()

        if 1 <= len(pos_probas):
            loss_1 = (neg_probas - pos_proba_min).mean()

        return auc_loss_coef * (loss_1 + loss_2) / 2

    def loss_method_4(logits, labels=None):
        if labels is None:
            return 0.0
        probas = logits[:, 1]
        pos_probas = labels * probas
        neg_probas = (1 - labels) * (1 - logits[:, 0])
        neg_proba_max = neg_probas.max().detach()
        pos_proba_min = pos_probas.min().detach()
        loss_1, loss_2 = 0.0, 0.0

        pos_probas = pos_probas[pos_probas < neg_proba_max]
        neg_probas = neg_probas[neg_probas > pos_proba_min]

        neg_proba_mean = neg_probas.mean()
        pos_proba_mean = pos_probas.mean()

        if 1 <= len(neg_probas):
            loss_2 = (neg_proba_mean - pos_probas).mean()

        if 1 <= len(pos_probas):
            loss_1 = (neg_probas - pos_proba_mean).mean()

        return auc_loss_coef * (loss_1 + loss_2) / 2
    
    def loss_method_5(logits, labels=None):
        if labels is None:
            return 0.0
        pos_probas = logits[:, 1][labels == 1]
        neg_probas = logits[:, 0][labels == 0]
        k1 = 0.4 # K1 <= 0.5 is necessary
        neg_rmse_labels = torch.rand_like(labels == 0) * k1
        pos_rmse_labels = (1 - k1) + torch.rand_like(labels == 1) * k1

        loss = (((neg_rmse_labels - neg_probas) ** 2) + ((pos_rmse_labels - pos_probas) ** 2)).mean()
        return auc_loss_coef * loss

    def loss_method_6(logits, labels=None):
        if labels is None:
            return 0.0
        pos_probas = logits[:, 1][labels == 1]
        neg_probas = logits[:, 0][labels == 0]
        k1 = 0.4
        k2 = 0.2
        k = k1 + torch.rand(1)[0] * k2
        neg_rmse_labels = torch.rand_like(labels == 0) * k
        pos_rmse_labels = k + torch.rand_like(labels == 1) * (1 - k)

        loss = (((neg_rmse_labels - neg_probas) ** 2) + ((pos_rmse_labels - pos_probas) ** 2)).mean()
        return auc_loss_coef * loss

    if auc_method == 1:
        return loss_method_1
    if auc_method == 2:
        return loss_method_2
    if auc_method == 3:
        return loss_method_3
    if auc_method == 4:
        return loss_method_4
    if auc_method == 5:
        return loss_method_5
    if auc_method == 6:
        return loss_method_6


def get_auc_dice_loss(n_classes, dice_loss_coef, auc_loss_coef, auc_method=1):
    dice_ll = get_dice_loss(n_classes, dice_loss_coef)
    auc_ll = get_auc_loss(n_classes, auc_loss_coef, auc_method)

    def loss(logits, labels=None):
        return dice_ll(logits, labels) + auc_ll(logits, labels)
    return loss


def calculate_auc_dice_loss(logits, labels, loss, auc_loss_coef, dice_loss_coef):
    binary = logits.size(1) == 2
    if binary:
        auc_loss = 0.0
        probas = logits[:, 1]
        if auc_loss_coef > 0:
            pos_probas = labels * probas
            neg_probas = (1 - labels) * probas

            loss_1, loss_2 = 0.0, 0.0
            neg_proba_max = neg_probas.max()
            pos_proba_min = pos_probas.min()
            loss_1 = F.leaky_relu(neg_probas - pos_proba_min).mean()
            loss_2 = F.leaky_relu(neg_proba_max - pos_probas).mean()

            pos_probas = pos_probas[pos_probas < neg_proba_max]
            neg_probas = neg_probas[neg_probas > pos_proba_min]

            num_entries_neg = max(1, int(len(neg_probas) / 4))
            num_entries_pos = max(1, int(len(pos_probas) / 4))
            if 1 <= num_entries_neg <= len(neg_probas):
                neg_probas_max = torch.topk(neg_probas, num_entries_neg, 0).values.mean()
                # neg_probas_max = neg_probas.max()
                loss_2 = (neg_probas_max - pos_probas).mean()


            if 1 <= num_entries_pos <= len(pos_probas):
                pos_probas_min = torch.topk(pos_probas, num_entries_pos, 0, largest=False).values.mean()
                # pos_probas_min = pos_probas.min()
                loss_1 = (neg_probas - pos_probas_min).mean()
            auc_loss = loss_1 + loss_2

        dice = dice_loss(probas, sigma=dice_sigma).mean()
        loss = (loss + auc_loss_coef * auc_loss + dice_loss_coef * dice) / (1 + auc_loss_coef + dice_loss_coef)

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
          collate_fn=my_collate,
          class_weights={0: 1, 1: 1.8}):
    if in_notebook():
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm as tqdm, trange

    if hasattr(dataset, "labels"):
        training_fold_labels = torch.tensor(dataset.labels)

    assert hasattr(dataset, "labels") or sampling_policy is None

    assert accumulation_steps >= 1 and type(accumulation_steps) == int
    _ = model.train()
    use_autocast = False
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        use_autocast = "cuda" in str(get_device())
    except:
        pass
    use_autocast = use_autocast and get_global("use_autocast")
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
        examples = len(dataset)
        divisor = 1
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                              shuffle=shuffle, num_workers=get_global("dataloader_workers"), pin_memory=True, sampler=sampler)

    train_losses = []
    learning_rates = []
    epochs = int(epochs * divisor)
    scheduler, update_in_batch, update_in_epoch = scheduler_init_fn(optimizer, epochs, batch_size, examples) if scheduler_init_fn is not None else (None, False, False)
    print("Autocast = ", use_autocast, "Epochs = ", epochs, "Divisor =", divisor, "Examples =", examples, "Batch Size = ", batch_size,)
    print("Training Samples = ", len(dataset), "Weighted Sampling = ", sampler is not None,
          "Num Batches = ", len(train_loader), "Accumulation steps = ", accumulation_steps)
    if len(train_loader) % accumulation_steps != 0:
        print("[WARN]: Number of training batches not divisible by accumulation steps, some training batches will be wasted due to this.")
    with trange(epochs) as epo:
        for epoc in epo:
            _ = model.train()
            optimizer.zero_grad()
            if update_in_epoch:
                scheduler.step()
            clean_memory()
            train_losses_cur_epoch = []
            loss_monitor = 0.0
            with tqdm(train_loader, "Batches") as data_batch:
                for batch_idx, batch in enumerate(data_batch):
                    if model_call_back is not None:
                        model_call_back(model, batch_idx, len(train_loader), epoc, epochs)
                    if use_autocast:
                        with autocast():
                            res = model(batch)
                            loss = res[-1]
                            loss = loss / accumulation_steps
                            loss_monitor += loss.cpu().detach().item()
                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()

                    else:
                        res = model(batch)
                        loss = res[-1]
                        loss = loss / accumulation_steps
                        loss_monitor += loss.cpu().detach().item()
                        loss.backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            optimizer.step()

                    if (batch_idx + 1) % accumulation_steps == 0:
                        train_losses.append(float(loss_monitor))
                        learning_rates.append(float(optimizer.param_groups[0]['lr']))
                        loss_monitor = 0.0
                        optimizer.zero_grad()
                    if update_in_batch:
                        scheduler.step()
                    train_losses_cur_epoch.append(float(loss.cpu().detach().item()) * accumulation_steps)
                    clean_memory()
            print("Epoch = ", epoc + 1, "Loss = %.6f" % np.mean(train_losses_cur_epoch), "LR = %.8f" % optimizer.param_groups[0]['lr'])
            if validation_strategy is not None:
                if (epoc + 1) in validation_strategy["validation_epochs"]:
                    vst, vsv = 0, 0
                    if "train" in validation_strategy:
                        vst, _ = validation_strategy["train"]["method"](*validation_strategy["train"]["args"], **validation_strategy["train"]["kwargs"])
                        vst = vst[-1]
                    if "val" in validation_strategy:
                        vsv, _ = validation_strategy["val"]["method"](*validation_strategy["val"]["args"],  **validation_strategy["val"]["kwargs"])
                        vsv = vsv[-1]
                    print("Epoch = ", epoc + 1, "Train = %.6f" % vst, "Val = %.6f" % vsv,)

    if plot:
        plot_loss_lr(train_losses, learning_rates)
    return train_losses, learning_rates


def train_for_augment_similarity(model, optimizer, scheduler_init_fn,
                                 batch_size, epochs, dataset,
                                 augment_method=lambda x: x,
                                 model_call_back=None, accumulation_steps=1,
                                 collate_fn=None,
                                 plot=False):
    import copy
    orig_model = copy.deepcopy(model).to(get_device())
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
    use_autocast = use_autocast and get_global("use_autocast")
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
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
            with tqdm(train_loader, "Batches") as data_batch:
                for batch_idx, batch in enumerate(data_batch):
                    if model_call_back is not None:
                        model_call_back(model, batch_idx, len(train_loader), epoc, epochs)
                    if type(batch) == list and len(batch) == 2:
                        augmented_batch = batch[1]
                        batch = batch[0]
                    else:
                        augmented_batch = augment_method(batch)
                    if use_autocast:
                        with autocast():
                            repr = model(augmented_batch)
                            with torch.no_grad():
                                orig_repr = orig_model(batch)
                            loss = ((repr - orig_repr)**2).mean()
                            loss = loss / accumulation_steps

                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            clean_memory()
                    else:
                        repr = model(augmented_batch)
                        with torch.no_grad():
                            orig_repr = orig_model(batch)
                        loss = ((repr - orig_repr) ** 2).mean()
                        loss = loss / accumulation_steps
                        loss.backward()
                        if (batch_idx + 1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            clean_memory()
                    if update_in_batch:
                        scheduler.step()
                    train_losses.append(float(loss.cpu().detach().item()))
                    train_losses_cur_epoch.append(float(loss.cpu().detach().item()))
                    learning_rates.append(float(optimizer.param_groups[0]['lr']))

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


def generate_predictions(model, batch_size, dataset,
                         prediction_iters=1, evaluate_in_train_mode=False,
                         collate_fn=my_collate):
    if evaluate_in_train_mode:
        _ = model.train()
    else:
        _ = model.eval()
    proba_list = []
    predictions_list = []
    all_probas_list = []
    clean_memory()
    from tqdm.auto import tqdm as tqdm, trange
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                             shuffle=False, num_workers=get_global("dataloader_workers"), pin_memory=True)

    use_autocast = False
    try:
        from torch.cuda.amp import autocast
        use_autocast = "cuda" in str(get_device())
    except:
        pass
    use_autocast = use_autocast and get_global("use_autocast")
    with torch.no_grad():
        clean_memory()
        logits_all = []

        for i in range(prediction_iters):
            labels_list = []
            logits_list = []
            with tqdm(test_loader, "Generate Predictions") as test_loader:
                for batch in test_loader:
                    if use_autocast:
                        with autocast():
                            logits, _, _, _ = model(batch)
                    else:
                        logits, _, _, _ = model(batch)
                    try:
                        batch = dict2sampleList(batch, device=get_device())
                        labels = batch["label"]
                    except:
                        labels = batch[-1]
                    labels_list.extend(labels.tolist() if type(labels) == torch.Tensor else labels)
                    logits = logits.cpu().detach()
                    logits_list.extend(logits.tolist())
            logits_all.append(logits_list)
        logits = torch.tensor(logits_all).mean(0)
        top_class = logits.max(dim=1).indices
        top_class = top_class.flatten().tolist()
        probas = logits[:, 1].tolist()
        all_probas = logits.tolist()
        all_probas_list.extend(all_probas)
        predictions_list.extend(top_class)
        proba_list.extend(probas)
        clean_memory()
    return proba_list, all_probas_list, predictions_list, labels_list


def validate(model, batch_size, dataset, collate_fn=my_collate, display_detail=False, prediction_iters=1, evaluate_in_train_mode=False,):
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    proba_list, all_probas_list, predictions_list, labels_list = generate_predictions(model, batch_size, dataset, collate_fn=collate_fn,
                                                                                      prediction_iters=prediction_iters,
                                                                                      evaluate_in_train_mode=evaluate_in_train_mode,)

    try:
        auc = roc_auc_score(labels_list, proba_list, multi_class="ovo", average="macro")
        map = average_precision_score(labels_list, proba_list)
    except:
        auc = 0
        map = 0
    # p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(labels_list, predictions_list, average="micro")
    prfs = precision_recall_fscore_support(labels_list, predictions_list, average=None, labels=[0, 1])
    acc = accuracy_score(labels_list, predictions_list)
    validation_scores = [map, acc, auc]
    if display_detail:
        few_preds = pd.DataFrame(np.random.permutation(list(zip(proba_list, all_probas_list, predictions_list, labels_list))), columns=["Proba", "Probas", "Preds", "Labels"])
        grouped_results = few_preds[["Proba", "Preds", "Labels"]]
        grouped_results = grouped_results.groupby(["Labels"])
        grouped_results = grouped_results[["Proba", "Preds"]]
        _ = grouped_results.agg(["min"])
        grouped_results = grouped_results.agg(["min", "max"])
        display(grouped_results)
        show_df = pd.concat((few_preds.head(5).reset_index(), few_preds.sample(5).reset_index(), few_preds.tail(5).reset_index()), 1).drop(columns=["index"])
        display(show_df)
        print("scores = ", dict(zip(["map", "acc", "auc"], ["%.4f" % v for v in validation_scores])))
    return validation_scores, prfs


def model_builder(model_class, model_params,
                  optimiser_class=torch.optim.AdamW, per_param_opts_fn=None,
                  optimiser_params=dict(lr=0.001, weight_decay=1e-5)):
    def builder(**kwargs):
        prams = dict(model_params)
        prams.update(kwargs)
        model = model_class(**prams)
        model.to(get_device())
        all_params = list(filter(lambda p: p.requires_grad, model.parameters()))

        if per_param_opts_fn is not None:
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            if type(per_param_opts_fn) == list:
                params_conf = per_param_opts_fn
            elif type(per_param_opts_fn) == dict:
                _ = group_wise_finetune(model, per_param_opts_fn)
                params_conf, _ = group_wise_lr(model, per_param_opts_fn)
            else:
                raise NotImplementedError()
            assert type(params_conf) == list
            assert len(params_conf) > 0
            assert all(["params" in p for p in params_conf])

            all_params = params_conf
        optimizer = None
        if len(all_params) > 0:
            optimizer = optimiser_class(all_params, **optimiser_params)
        return model, optimizer

    return builder


def convert_dataframe_to_dataset(df, metadata, train=True, **kwargs):
    from functools import partial
    text = list(df.text)
    labels = df["label"].values if "label" in df else None
    sample_weights = df["sample_weights"].values if "sample_weights" in df else None
    assert "id" in df.columns or "ID" in df.columns
    id_name = "id" if "id" in df.columns else "ID"
    ids = list(df[id_name])
    ds = TextImageDataset(ids, text, list(df.img), labels, sample_weights,
                          text_transform=metadata["train_text_transform"] if train else metadata["test_text_transform"],
                          torchvision_image_transform=metadata["train_torchvision_image_transform"] if train else metadata["test_torchvision_image_transform"],
                          torchvision_pre_image_transform=metadata["train_torchvision_pre_image_transform"] if train else metadata["test_torchvision_pre_image_transform"],
                          image_transform=metadata["train_image_transform"] if train else metadata["test_image_transform"],
                          cache_images=metadata["cache_images"], use_images=metadata["use_images"],
                          keep_original_text=metadata["keep_original_text"], keep_original_image=metadata["keep_original_image"],
                          keep_processed_image=metadata["keep_processed_image"], keep_torchvision_image=metadata["keep_torchvision_image"],
                          mixup_config=metadata["train_mixup_config"] if train else None,  **kwargs)
    return ds


def random_split_for_augmented_dataset(datadict, n_splits=5, random_state=None, uniform_sample_binary_labels=True):
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold, KFold
    metadata = datadict["metadata"]
    train = datadict["train"]
    if uniform_sample_binary_labels and len(set(train.label)) == 2:
        # TODO: This code is specific to FB competition since it assumes that zeros are more than ones
        ones = train[train["label"] == 1]
        zeros = train[train["label"] == 0]
        kf1, kf2 = KFold(n_splits=5, shuffle=True), KFold(n_splits=5, shuffle=True)
        for (ones_train_idx, ones_test_idx), (zeros_train_idx, zeros_test_idx) in zip(kf1.split(ones), kf2.split(zeros)):
            ones_train = ones.iloc[ones_train_idx]
            ones_test = ones.iloc[ones_test_idx]
            zeros_train = pd.concat((zeros.iloc[zeros_train_idx], zeros.iloc[zeros_test_idx[len(ones_test_idx):]]))
            zeros_test = zeros.iloc[zeros_test_idx[:len(ones_test_idx)]]
            train_split = pd.concat((ones_train, zeros_train)).sample(frac=1.0)
            test_split = pd.concat((ones_test, zeros_test)).sample(frac=1.0)
            print("Doing Special split for FB", "\n", "Train Labels =", train_split.label.value_counts(),
                  "Test Labels =", test_split.label.value_counts())
            assert len(set(ones_train["id"]).intersection(set(ones_test["id"]))) == 0
            assert len(set(zeros_train["id"]).intersection(set(zeros_test["id"]))) == 0
            assert len(set(train_split["id"]).intersection(set(test_split["id"]))) == 0
            train_set = convert_dataframe_to_dataset(train_split, metadata, True)
            train_test_set = convert_dataframe_to_dataset(train_split, metadata, False, cached_images=train_set.images)
            yield (train_set,
                   train_test_set,
                   convert_dataframe_to_dataset(test_split, metadata, False))

    else:
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        # print("TRAIN Sizes =",train.shape, "\n", train.label.value_counts())
        # skf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        for train_idx, test_idx in skf.split(train, train.label):
            train_split = train.iloc[train_idx]
            test_split = train.iloc[test_idx]
            assert len(set(train_split["id"]).intersection(set(test_split["id"]))) == 0
            # print("Train Test Split sizes =","\n",train_split.label.value_counts(),"\n",test_split.label.value_counts())
            train_set = convert_dataframe_to_dataset(train_split, metadata, True)
            train_test_set = convert_dataframe_to_dataset(train_split, metadata, False, cached_images=train_set.images)
            yield (train_set,
                   train_test_set,
                   convert_dataframe_to_dataset(test_split, metadata, False))


def train_validate_ntimes(model_fn, data, batch_size, epochs,
                          accumulation_steps=1,
                          kfold=False,
                          scheduler_init_fn=None, model_call_back=None,
                          random_state=None, validation_epochs=None, show_model_stats=False,
                          sampling_policy=None,
                          class_weights=None,
                          prediction_iters=1,
                          evaluate_in_train_mode=False
                          ):
    from tqdm import tqdm
    getattr(tqdm, '_instances', {}).clear()
    from tqdm.auto import tqdm as tqdm, trange
    results_list = []
    prfs_list = []
    index = ["map", "accuracy", "auc"]
    test_dev = data["metadata"]["test_dev"]
    assert not (test_dev and kfold)

    if test_dev:
        trin = data["train"]
        test = data["dev"]
        metadata = data["metadata"]
        train_set = convert_dataframe_to_dataset(trin, metadata, True)
        train_test_set = convert_dataframe_to_dataset(trin, metadata, False, cached_images=train_set.images)
        folds = [(train_set,
                  train_test_set,
                  convert_dataframe_to_dataset(test, metadata, False))]
    else:
        folds = random_split_for_augmented_dataset(data, random_state=random_state)

    for training_fold_dataset, training_test_dataset, testing_fold_dataset in folds:
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
                                   train=dict(method=validate, args=[model, batch_size, training_test_dataset], kwargs=dict(display_detail=False)),
                                   val=dict(method=validate, args=[model, batch_size, testing_fold_dataset], kwargs=dict(display_detail=True,
                                                                                                                         prediction_iters=prediction_iters,
                                                                                                                         evaluate_in_train_mode=evaluate_in_train_mode)))
        validation_strategy = validation_strategy if validation_epochs is not None else None
        train_losses, learning_rates = train(model, optimizer, scheduler_init_fn, batch_size, epochs, training_fold_dataset, model_call_back, accumulation_steps,
                                             validation_strategy, plot=not kfold, sampling_policy=sampling_policy, class_weights=class_weights)

        validation_scores, prfs_val = validate(model, batch_size, testing_fold_dataset, display_detail=True)
        train_scores, prfs_train = validate(model, batch_size, training_test_dataset, display_detail=False, prediction_iters=prediction_iters,
                                                                                                            evaluate_in_train_mode=evaluate_in_train_mode)
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

