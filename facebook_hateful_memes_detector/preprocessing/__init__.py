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
from ..utils import read_json_lines_into_df

import torch as th
import math
import os
import re
import contractions
from pycontractions import Contractions
from torch.utils.data.sampler import WeightedRandomSampler


def clean_text(text):
    # https://stackoverflow.com/questions/6202549/word-tokenization-using-python-regular-expressions
    # https://stackoverflow.com/questions/44263446/python-regex-to-add-space-after-dot-or-comma/44263500
    EMPTY = ' '
    assert text is not None

    text = re.sub(r'([A-Z][a-z]+(?=[A-Z]))', r'\1 ', text)
    text = re.sub(r'(?<=[.,;])(?=[^\s0-9])', ' ', text)
    text = contractions.fix(text)
    text = text.lower()

    text = text.replace("'", " ").replace('"', " ")
    text = text.replace("\n", " ").replace("(", " ").replace(")", " ").replace("\r", " ").replace("\t", " ")

    text = re.sub('<pre><code>.*?</code></pre>', EMPTY, text)
    text = re.sub('<code>.*?</code>', EMPTY, text)

    def replace_link(match):
        return EMPTY if re.match('[a-z]+://', match.group(1)) else match.group(1)

    text = re.sub('<a[^>]+>(.*)</a>', replace_link, text)
    text = re.sub('<.*?>', EMPTY, text)
    text = re.sub(r"[^A-Za-z0-9.!$\'? ]+", '', text)
    text = " ".join([t.strip() for t in text.split()])
    return text

import os
import random


class TextAugment:
    def __init__(self, proba, choice_probas, count=1):
        self.proba = proba

        self.count = count
        import nlpaug.augmenter.char as nac
        import nlpaug.augmenter.word as naw
        import nlpaug.augmenter.sentence as nas
        import nlpaug.flow as naf
        from nlpaug.util import Action
        self.keyboard_aug = nac.KeyboardAug(aug_char_min=1, aug_char_max=3, aug_word_min=1, aug_word_max=3, include_special_char=False, include_numeric=False, include_upper_case=False)
        self.ocr_aug = nac.OcrAug(aug_char_min=1, aug_char_max=3, aug_word_min=1, aug_word_max=3)
        self.char_insert = nac.RandomCharAug(action="insert", aug_char_min=1, aug_char_max=3, aug_word_min=1, aug_word_max=3, include_numeric=False, include_upper_case=False)
        self.char_substitute = nac.RandomCharAug(action="substitute", aug_char_min=1, aug_char_max=3, aug_word_min=1,
                                             aug_word_max=3, include_numeric=False, include_upper_case=False)
        self.char_swap = nac.RandomCharAug(action="swap", aug_char_min=1, aug_char_max=3, aug_word_min=1,
                                             aug_word_max=3, include_numeric=False, include_upper_case=False)
        self.char_delete = nac.RandomCharAug(action="delete", aug_char_min=1, aug_char_max=3, aug_word_min=1,
                                             aug_word_max=3, include_numeric=False, include_upper_case=False)
        self.word_insert = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action='insert', temperature=0.5, top_k=20, aug_min=1, aug_max=3, optimize=True)
        self.word_substitute = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action='substitute', temperature=0.5, top_k=20, aug_min=1, aug_max=3, optimize=True)
        self.augments = {"keyboard": self.keyboard_aug, "ocr": self.ocr_aug, "char_insert": self.char_insert, "char_substitute": self.char_substitute,
                         "char_swap": self.char_swap, "char_delete": self.char_delete, "word_insert": self.word_insert, "word_substitute": self.word_substitute}
        self.augs = ["keyboard", "ocr", "char_insert", "char_substitute", "char_swap", "char_delete", "word_insert", "word_substitute"]
        self.choice_probas = np.array([choice_probas[c] if c in choice_probas else 0.0 for c in self.augs])
        self.choice_probas = self.choice_probas / np.linalg.norm(self.choice_probas, ord=1)

    def __call__(self, text):
        if random.random() < self.proba:
            augs = np.random.choice(self.augs, self.count, replace=False, p=self.choice_probas)
            for aug in augs:
                text = self.augments[aug].augment(text)
        return text






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
                 cache_images: bool = True, use_images: bool = True, dev: bool = False):
    use_dev = dev
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

    ts = (dev_text, dev_img, dev_labels) if use_dev else (train_text, train_img, train_labels)
    dataset = TextImageDataset(*ts,
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

    rd = dict(train=dataset, test=test_ds, dev=deve_ds, test_df=test, dev_df=dev, train_df=train)
    return rd


def make_weights_for_balanced_classes(labels, weight_per_class: Dict = None):
    labels = labels.numpy()
    from collections import Counter
    count = Counter(labels)
    N = len(labels)
    if weight_per_class is None:
        weight_per_class = {clas: N / float(occ) for clas, occ in count.items()}
    weight = [weight_per_class[label] for label in labels]
    return torch.DoubleTensor(weight)


class TextImageDataset(Dataset):
    def __init__(self, texts: List[str], image_locations: List[str], labels: torch.Tensor=None,
                 text_transform=None, image_transform=None, cache_images: bool = True, use_images: bool = True):
        self.texts = [clean_text(text) for text in texts]
        self.image_locations = image_locations
        self.is_transform = False
        if use_images:
            self.images = {l: Image.open(l).convert('RGB') for l in image_locations} if cache_images else dict()
        self.labels = labels
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.use_images = use_images

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item] if self.labels is not None else 0
        if self.text_transform is not None and self.is_transform:
            text = self.text_transform(text)
        if self.use_images:
            l = self.image_locations[item]
            image = self.images.get(l)
            if image is None:
                image = Image.open(l).convert('RGB')
            image = self.image_transform(image) if self.image_transform is not None else image
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


