import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
import numpy as np
import jsonlines
import abc
from typing import List, Tuple, Dict, Set, Union
from PIL import Image
from ..utils import clean_text, read_json_lines_into_df

import torch as th
import math
import os


def get_basic_image_transforms():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


def my_collate(batch):
    text = [item[0] for item in batch]
    image = torch.stack([item[1] for item in batch])
    label = [item[2] for item in batch]
    label = torch.LongTensor(label) if label[0] is not None else label
    return [text, image, label]


def get_datasets(data_dir, train_text_transform=None, train_image_transform=None,
                 test_text_transform=None, test_image_transform=None,
                 cache_images: bool = True, use_images: bool = True):
    from functools import partial
    joiner = partial(os.path.join, data_dir)
    dev = read_json_lines_into_df(joiner('dev.jsonl'))
    train = read_json_lines_into_df(joiner('train.jsonl'))
    test = read_json_lines_into_df(joiner('test.jsonl'))

    train_text = list(train.text)
    dev_text = list(dev.text)
    test_text = list(test.text)

    train_img = list(map(joiner, train.img))
    test_img = list(map(joiner, test.img))
    dev_img = list(map(joiner, dev.img))

    train_labels = torch.tensor(train.label)
    dev_labels = torch.tensor(dev.label)

    train_text = np.array(train_text)
    dev_text = np.array(dev_text)
    test_text = np.array(test_text)

    train_img = np.array(train_img)
    test_img = np.array(test_img)
    dev_img = np.array(dev_img)

    dataset = TextImageDataset(train_text, train_img, train_labels,
                               text_transform=train_text_transform,
                               image_transform=train_image_transform,
                               cache_images=cache_images, use_images=use_images)

    test_ds = TextImageDataset(test_text, test_img, None,
                               text_transform=test_text_transform,
                               image_transform=test_image_transform,
                               cache_images=cache_images, use_images=use_images)

    deve_ds = TextImageDataset(dev_text, dev_img, dev_labels,
                               text_transform=train_text_transform,
                               image_transform=train_image_transform,
                               cache_images=cache_images, use_images=use_images)

    rd = dict(train=dataset, test=test_ds, dev=deve_ds, test_df=test)
    return rd


class StratifiedSampler(torch.utils.data.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        super(StratifiedSampler, self).__init__(class_vector)
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = th.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


class TextImageDataset(Dataset):
    def __init__(self, texts: List[str], image_locations: List[str], labels: torch.Tensor=None,
                 text_transform=None, image_transform=None, cache_images: bool = True, use_images: bool = True):
        self.texts = texts
        self.image_locations = image_locations
        if use_images:
            self.images = {l: Image.open(l).convert('RGB') for l in image_locations} if cache_images else dict()
        self.labels = labels
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.use_images = use_images

    def __getitem__(self, item):
        text = self.texts[item]
        text = clean_text(text)
        label = self.labels[item] if self.labels is not None else 0
        text = self.text_transform(text) if self.text_transform is not None else text
        if self.use_images:
            l = self.image_locations[item]
            image = self.images.get(l)
            if image is None:
                image = Image.open(l).convert('RGB')
            image = self.image_transform(image.copy()) if self.image_transform is not None else image
            return text, image, label
        else:
            return text, torch.tensor(0), label

    def __len__(self):
        return len(self.texts)

    def show(self, item):
        l = self.image_locations[item]
        image = Image.open(l)
        text = self.texts[item]
        label = self.labels[item] if self.labels is not None else None
        print(text, "|", "Label = ", label)
        image.show()
        return image


