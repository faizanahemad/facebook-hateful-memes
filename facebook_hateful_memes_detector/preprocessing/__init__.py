import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
import numpy as np
import jsonlines
import abc
from typing import List, Tuple, Dict, Set, Union, Callable
from PIL import Image
from ..utils import read_json_lines_into_df, clean_memory

import torch as th
import math
import os
import re
import contractions
from pycontractions import Contractions
from torch.utils.data.sampler import WeightedRandomSampler
from mmf.common.sample import Sample, SampleList
from mmf.common.batch_collator import BatchCollator
import torchvision
import random


class DefinedRotation(torchvision.transforms.RandomRotation):
    def __init__(self, degrees):
        super().__init__(degrees)

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """

        angle = random.sample(list(degrees), k=1)[0]

        return angle


class HalfSwap:
    def __call__(self, image):
        arr = np.array(image)
        shape = list(arr.shape)
        shape[0] = (shape[0] - 1) if shape[0] % 2 == 1 else shape[0]
        shape[1] = (shape[1] - 1) if shape[1] % 2 == 1 else shape[1]
        arr = arr[:shape[0], :shape[1]]
        height_half = int(shape[0] / 2)
        width_half = int(shape[1] / 2)
        if random.random() < 0.5:
            arr[:height_half], arr[height_half:] = arr[height_half:], arr[:height_half]
        else:
            arr[:, :width_half], arr[:, width_half:] = arr[:, width_half:], arr[:, :width_half]

        return Image.fromarray(arr)


class QuadrantCut:
    def __call__(self, image):
        arr = np.array(image)  # H, W, C PIL image
        mean = 110  # mean = np.mean(arr)
        shape = arr.shape
        x_half = int(shape[0] / 2)
        y_half = int(shape[1] / 2)

        x_third = int(shape[0] / 2.5)
        y_third = int(shape[1] / 2.5)
        x_2third = shape[0] - x_third
        y_2third = shape[1] - y_third

        choice = random.randint(1, 11)
        if choice == 1:
            arr[:x_half, :y_half] = mean
        if choice == 2:
            arr[:x_half, y_half:] = mean
        if choice == 3:
            arr[x_half:, y_half:] = mean
        if choice == 4:
            arr[x_half:, :y_half] = mean

        if choice == 5:
            arr[:x_half, :] = mean
        if choice == 6:
            arr[x_half:, :] = mean
        if choice == 7:
            arr[:, y_half:] = mean
        if choice == 8:
            arr[:, :y_half] = mean

        if choice == 9:
            arr[x_third:x_2third, :] = mean
        if choice == 10:
            arr[:, y_third:y_2third] = mean
        if choice == 11:
            arr[x_third:x_2third, y_third:y_2third] = mean

        return Image.fromarray(arr)


class DefinedColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        from torchvision.transforms import Lambda, Compose

        if brightness is not None:
            brightness_factor = random.sample(list(brightness), k=1)[0]
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.sample(list(contrast), k=1)[0]
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.sample(list(saturation), k=1)[0]
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.sample(list(hue), k=1)[0]
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform


class DefinedRandomPerspective(torchvision.transforms.RandomPerspective):
    def __init__(self, distortion_scale=0.5, interpolation=Image.BICUBIC, fill=100):
        super().__init__(distortion_scale=distortion_scale, p=1.0, interpolation=interpolation, fill=fill)

    @staticmethod
    def get_params(width, height, distortion_scale):
        import random
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width : width of the image.
            height : height of the image.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.sample((0, int(distortion_scale * half_width)), k=1)[0],
                   random.sample((0, int(distortion_scale * half_height)), k=1)[0])

        topright = (random.sample((width - int(distortion_scale * half_width) - 1, width - 1), k=1)[0],
                    random.sample((0, int(distortion_scale * half_height)), k=1)[0]
                    )
        botright = (random.sample((width - int(distortion_scale * half_width) - 1, width - 1), k=1)[0],
                    random.sample((height - int(distortion_scale * half_height) - 1, height - 1), k=1)[0]
                    )
        botleft = (random.sample((0, int(distortion_scale * half_width)), k=1)[0],
                   random.sample((height - int(distortion_scale * half_height) - 1, height - 1), k=1)[0]
                   )
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints


class DefinedAffine(torchvision.transforms.RandomAffine):
    def __init__(self, degrees, translate=None, scale=None, shear=None,):
        super().__init__(degrees, translate, scale, shear, fillcolor=140)

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = random.sample(list(degrees), k=1)[0]
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            t1 = random.sample((-max_dx, 0.0, max_dx), k=1)[0]
            if t1 == 0:
                t2 = random.sample((-max_dy, max_dy), k=1)[0]
            else:
                t2 = random.sample((-max_dy, 0.0, max_dy), k=1)[0]
            translations = (t1, t2)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.sample((scale_ranges[0], 1.0, scale_ranges[1]), k=1)[0]
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.sample((shears[0], 0., shears[1]), k=1)[0], 0.]
            elif len(shears) == 4:
                shear = [random.sample((shears[0], 0., shears[1]), k=1)[0],
                         random.sample((shears[2], 0., shears[3]), k=1)[0]]
        else:
            shear = 0.0
        return angle, translations, scale, shear


class ImageAugment:
    def __init__(self, count_proba: List[float], augs_dict: Dict[str, Callable], choice_probas: Dict[str, float]):
        self.count_proba = count_proba
        assert 1 - 1e-6 <= sum(count_proba) <= 1 + 1e-6
        assert len(count_proba) >= 1
        if choice_probas == "uniform":
            adl = len(augs_dict)
            choice_probas = {k: 1.0/adl for k, v in augs_dict.items()}
        assert (len(count_proba) - 1) < sum([v > 0 for v in choice_probas.values()])
        choice_probas = {k: v for k, v in choice_probas.items() if v > 0}
        assert set(choice_probas.keys()).issubset(augs_dict.keys())
        self.augs = list(augs_dict.keys())
        choices_arr = np.array([choice_probas[c] if c in choice_probas else 0.0 for c in self.augs])
        self.choice_probas = choices_arr / np.linalg.norm(choices_arr, ord=1)
        self.augments = augs_dict

    def __call__(self, image):
        count = np.random.choice(list(range(len(self.count_proba))), 1, replace=False, p=self.count_proba)[0]
        augs = np.random.choice(self.augs, count, replace=False, p=self.choice_probas)
        for aug in augs:
            try:
                image = self.augments[aug](image)

            except Exception as e:
                print("Exception for: ", aug, "|", "|", augs, e)
        return image


def clean_text(text):
    # https://stackoverflow.com/questions/6202549/word-tokenization-using-python-regular-expressions
    # https://stackoverflow.com/questions/44263446/python-regex-to-add-space-after-dot-or-comma/44263500
    EMPTY = ' '
    assert text is not None
    text = re.sub(r'([A-Z][a-z]+(?=[A-Z]))', r'\1 ', text)
    text = re.sub(r'(?<=[.,;!])(?=[^\s0-9])', ' ', text)
    text = re.sub('[ ]+', ' ', text)

    def replace_link(match):
        return EMPTY if re.match('[a-z]+://', match.group(1)) else match.group(1)

    text = re.sub('<a[^>]*>(.*)</a>', replace_link, text)
    text = re.sub('<.*?>', EMPTY, text)
    text = re.sub('\[.*?\]', EMPTY, text)
    text = contractions.fix(text)
    text = text.lower()

    text = text.replace("'", " ").replace('"', " ")
    text = text.replace("\n", " ").replace("(", " ").replace(")", " ").replace("\r", " ").replace("\t", " ")

    text = re.sub('<pre><code>.*?</code></pre>', EMPTY, text)
    text = re.sub('<code>.*?</code>', EMPTY, text)

    text = re.sub(r"[^A-Za-z0-9.!$,;\'? ]+", EMPTY, text)
    text = " ".join([t.strip() for t in text.split()])
    return text


import os
import random
import fasttext
import gensim.downloader as api
from nltk import sent_tokenize
from gensim.models.fasttext import load_facebook_model
from nltk.corpus import stopwords


class TextAugment:
    def __init__(self, count_proba: List[float], choice_probas: Dict[str, float], fasttext_file: str = None):
        self.count_proba = count_proba
        assert 1 - 1e-6 <= sum(count_proba) <= 1 + 1e-6
        assert len(count_proba) >= 1
        assert (len(count_proba) - 1) < sum([v > 0 for v in choice_probas.values()])

        import nlpaug.augmenter.char as nac
        import nlpaug.augmenter.word as naw
        from gensim.similarities.index import AnnoyIndexer

        def gibberish_insert(text):
            words = text.split()
            idx = random.randint(0, len(words) - 1)
            sw = "".join(random.sample("abcdefghijklmnopqrstuvwxyz.!", random.randint(3, 15)))
            words = words[:idx] + [sw] + words[idx:]
            return " ".join(words)

        def part_select(text, sp=0.6, lp=0.9):
            splits = text.split()
            if len(splits) <= 2:
                return text
            actual_len = random.randint(int(len(splits) * sp), int(len(splits) * lp))
            if actual_len == len(splits):
                return text
            left_length = len(splits) - actual_len
            start = random.randint(0, left_length)
            return " ".join(splits[start:start + actual_len])

        def one_third_cut(text):
            return part_select(text, 0.666, 0.666)

        def half_cut(text):
            return part_select(text, 0.5, 0.5)

        def sentence_shuffle(text):
            sents = sent_tokenize(text)
            random.shuffle(sents)
            return " ".join(sents)

        def text_rotate(text):
            words = text.split()
            if len(words) <= 2:
                return text
            rotate = random.randint(1, int(len(words)/2))
            words = words[rotate:] + words[:rotate]
            return " ".join(words)

        stopwords_list = stopwords.words("english")

        def stopword_insert(text):
            words = text.split()
            idx = random.randint(0, len(words) - 1)
            sw = random.sample(stopwords_list, 1)[0]
            words = words[:idx] + [sw] + words[idx:]
            return " ".join(words)

        def word_join(text):
            words = text.split()
            if len(words) <= 2:
                return text
            idx = random.randint(0, len(words) - 2)
            w1 = words[idx] + words[idx + 1]
            words = words[:idx] + [w1] + words[idx + 1:]
            return " ".join(words)

        def word_cutout(text):
            words = text.split()
            lwi = [i for i, w in enumerate(words) if len(w) >= 4]
            if len(lwi) <= 2:
                return text
            cut_idx = random.sample(lwi, 1)[0]
            words = words[:cut_idx] + words[cut_idx + 1:]
            return " ".join(words)

        self.augs = ["keyboard", "ocr", "char_insert", "char_substitute", "char_swap", "char_delete",
                     "word_insert", "word_substitute", "w2v_insert", "w2v_substitute", "text_rotate",
                     "stopword_insert", "word_join", "word_cutout",
                     "fasttext", "glove_twitter", "glove_wiki", "word2vec", "gibberish_insert",
                     "synonym", "split", "sentence_shuffle", "one_third_cut", "half_cut", "part_select"]
        assert len(set(list(choice_probas.keys())) - set(self.augs)) == 0
        self.augments = dict()
        self.indexes = dict()
        for k, v in choice_probas.items():
            if v <= 0:
                continue
            if k == "part_select":
                self.augments["part_select"] = part_select
            if k == "gibberish_insert":
                self.augments["gibberish_insert"] = gibberish_insert
            if k == "stopword_insert":
                self.augments["stopword_insert"] = stopword_insert
            if k == "word_join":
                self.augments["word_join"] = word_join
            if k == "word_cutout":
                self.augments["word_cutout"] = word_cutout
            if k == "text_rotate":
                self.augments["text_rotate"] = text_rotate
            if k == "sentence_shuffle":
                self.augments["sentence_shuffle"] = sentence_shuffle
            if k == "one_third_cut":
                self.augments["one_third_cut"] = one_third_cut
            if k == "half_cut":
                self.augments["half_cut"] = half_cut
            if k == "synonym":
                self.augments["synonym"] = naw.SynonymAug(aug_src='ppdb', model_path='ppdb-2.0-s-all', aug_max=1)
            if k == "split":
                self.augments["split"] = naw.SplitAug(aug_max=1, min_char=6,)

            if k == "fasttext":
                assert fasttext_file is not None
                self.augments["fasttext"] = load_facebook_model(fasttext_file)
                self.indexes["fasttext"] = AnnoyIndexer(self.augments["fasttext"], 8)
            if k == "word2vec":
                self.augments["word2vec"] = api.load("word2vec-google-news-300")
                self.indexes["word2vec"] = AnnoyIndexer(self.augments["word2vec"], 8)
            if k == "glove_twitter":
                self.augments["glove_twitter"] = api.load("glove-twitter-100")
                self.indexes["glove_twitter"] = AnnoyIndexer(self.augments["glove_twitter"], 8)
            if k == "glove_wiki":
                self.augments["glove_wiki"] = api.load("glove-wiki-gigaword-100")
                self.indexes["glove_wiki"] = AnnoyIndexer(self.augments["glove_wiki"], 8)

            if k == "keyboard":
                self.augments["keyboard"] = nac.KeyboardAug(aug_char_min=1, aug_char_max=3, aug_word_min=1,
                                                            aug_word_max=3, include_special_char=False,
                                                            include_numeric=False, include_upper_case=False)
            if k == "ocr":
                self.augments["ocr"] = nac.OcrAug(aug_char_min=1, aug_char_max=3, aug_word_min=1, aug_word_max=3, min_char=3)

            if k == "char_insert":
                self.augments["char_insert"] = nac.RandomCharAug(action="insert", aug_char_min=1, aug_char_max=2,
                                                                 aug_word_min=1, aug_word_max=3, include_numeric=False,
                                                                 include_upper_case=False)
            if k == "char_substitute":
                self.augments["char_substitute"] = nac.RandomCharAug(action="substitute", aug_char_min=1,
                                                                     aug_char_max=2, aug_word_min=1,
                                                                     aug_word_max=3, include_numeric=False,
                                                                     include_upper_case=False)
            if k == "char_swap":
                self.augments["char_swap"] = nac.RandomCharAug(action="swap", aug_char_min=1, aug_char_max=1,
                                                               aug_word_min=1,
                                                               aug_word_max=3, include_numeric=False,
                                                               include_upper_case=False)
            if k == "char_delete":
                self.augments["char_delete"] = nac.RandomCharAug(action="delete", aug_char_min=1, aug_char_max=1,
                                                                 aug_word_min=1,
                                                                 aug_word_max=3, include_numeric=False,
                                                                 include_upper_case=False)

            if k == "word_insert":
                self.augments["word_insert"] = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased',
                                                                         action='insert', temperature=0.5, top_k=20,
                                                                         aug_min=1, aug_max=1, optimize=True)
            if k == "word_substitute":
                self.augments["word_substitute"] = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased',
                                                                             action='substitute', temperature=0.5,
                                                                             top_k=20, aug_min=1, aug_max=1,
                                                                             optimize=True)
            if k == "w2v_insert":
                self.augments["w2v_insert"] = naw.WordEmbsAug(model_type='word2vec',
                                                              model_path='GoogleNews-vectors-negative300.bin',
                                                              action="insert", aug_min=1, aug_max=1, top_k=10, )
            if k == "w2v_substitute":
                self.augments["w2v_substitute"] = naw.WordEmbsAug(model_type='word2vec',
                                                                  model_path='GoogleNews-vectors-negative300.bin',
                                                                  action="substitute", aug_min=1, aug_max=1, top_k=10, )

        choices_arr = np.array([choice_probas[c] if c in choice_probas else 0.0 for c in self.augs])
        self.choice_probas = choices_arr / np.linalg.norm(choices_arr, ord=1)

    def __fasttext_replace__(self, tm, indexer, text):
        tokens = text.split()
        t_2_i = {w: i for i, w in enumerate(tokens) if len(w) >= 4}
        if len(t_2_i) <= 2:
            return text
        sampled = random.sample(list(t_2_i.keys()), k=1)[0]
        sampled_idx = t_2_i[sampled]
        # candidates = [w for d, w in self.augments["fasttext"].get_nearest_neighbors(sampled, 10)]
        candidates = [w for w, d in tm.most_similar(sampled, topn=10, indexer=indexer)][1:]
        replacement = random.sample(candidates, k=1)[0]
        tokens[sampled_idx] = replacement
        return " ".join(tokens)

    def __w2v_replace__(self, tm, indexer, text):
        tokens = text.split()
        t_2_i = {w: i for i, w in enumerate(tokens) if len(w) >= 4}
        if len(t_2_i) <= 2:
            return text
        success = False
        repeat_count = 0
        while not success and repeat_count <= 10:
            repeat_count += 1
            sampled = random.sample(list(t_2_i.keys()), k=1)[0]
            if sampled in tm:
                candidates = [w for w, d in tm.most_similar(sampled, topn=10, indexer=indexer)][1:]
                success = True
        if not success:
            return text
        sampled_idx = t_2_i[sampled]
        replacement = random.sample(candidates, k=1)[0]
        tokens[sampled_idx] = replacement
        return " ".join(tokens)

    def __call__(self, text):
        count = np.random.choice(list(range(len(self.count_proba))), 1, replace=False, p=self.count_proba)[0]
        augs = np.random.choice(self.augs, count, replace=False, p=self.choice_probas)
        for aug in augs:
            try:
                if aug == "fasttext":
                    text = self.__fasttext_replace__(self.augments[aug], self.indexes[aug], text)
                elif aug in ["glove_twitter", "glove_wiki", "word2vec"]:
                    text = self.__w2v_replace__(self.augments[aug], self.indexes[aug], text)
                elif callable(self.augments[aug]):
                    text = self.augments[aug](text)
                elif hasattr(self.augments[aug], "augment"):
                    text = self.augments[aug].augment(text)
                else:
                    raise ValueError()
            except Exception as e:
                print("Exception for: ", aug, "|", text, "|", augs, e)
        return text


def get_image2torchvision_transforms():
    preprocess = transforms.Compose([
        transforms.Resize(352),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


def get_csv_datasets(train_file, test_file, image_dir, train_text_transform=None, train_image_transform=None,
                     train_torchvision_image_transform=None, test_torchvision_image_transform=None,
                     train_torchvision_pre_image_transform=None, test_torchvision_pre_image_transform=None,
                     test_text_transform=None, test_image_transform=None,
                     cache_images: bool = True, use_images: bool = True, dev: bool = False,
                     keep_original_text: bool = False, keep_original_image: bool = False,
                     train_mixup_config=None, test_mixup_config=None,
                     keep_processed_image: bool = False, keep_torchvision_image: bool = False):
    from functools import partial
    use_dev = dev
    joiner = lambda img: os.path.join(image_dir, img) if img is not None and type(img) == str and img != "nan" else None
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    dev = train.sample(frac=0.1)

    dev["img"] = list(map(joiner, dev.img))
    train["img"] = list(map(joiner, train.img))
    test["img"] = list(map(joiner, test.img))

    rd = dict(train=train, test=test, dev=dev,
              metadata=dict(cache_images=cache_images, use_images=use_images, dev=use_dev,
                            keep_original_text=keep_original_text, keep_original_image=keep_original_image,
                            keep_processed_image=keep_processed_image, keep_torchvision_image=keep_torchvision_image,
                            train_text_transform=train_text_transform, train_image_transform=train_image_transform,
                            train_torchvision_image_transform=train_torchvision_image_transform,
                            train_torchvision_pre_image_transform=train_torchvision_pre_image_transform,
                            train_mixup_config=train_mixup_config, test_mixup_config=test_mixup_config,
                            test_torchvision_pre_image_transform=test_torchvision_pre_image_transform,
                            test_torchvision_image_transform=test_torchvision_image_transform,
                            test_text_transform=test_text_transform, test_image_transform=test_image_transform))
    return rd


def get_datasets(data_dir, train_text_transform=None, train_image_transform=None,
                 train_torchvision_image_transform=None, test_torchvision_image_transform=None,
                 train_torchvision_pre_image_transform=None, test_torchvision_pre_image_transform=None,
                 test_text_transform=None, test_image_transform=None,
                 train_mixup_config=None, test_mixup_config=None,
                 cache_images: bool = True, use_images: bool = True, dev: bool = False, test_dev: bool = True,
                 keep_original_text: bool = False, keep_original_image: bool = False,
                 keep_processed_image: bool = False, keep_torchvision_image: bool = False):
    use_dev = dev
    from functools import partial
    joiner = partial(os.path.join, data_dir)
    dev = read_json_lines_into_df(joiner('dev.jsonl'))
    train = read_json_lines_into_df(joiner('train.jsonl'))
    test = read_json_lines_into_df(joiner('test.jsonl'))

    dev["img"] = list(map(joiner, dev.img))
    train["img"] = list(map(joiner, train.img))
    test["img"] = list(map(joiner, test.img))

    submission_format = pd.read_csv(joiner("submission_format.csv"))
    # TODO: Fold in dev into train
    if not test_dev:
        train = pd.concat((train, dev))
    if use_dev:
        train = dev

    rd = dict(train=train, test=test, dev=dev,
              submission_format=submission_format,
              metadata=dict(cache_images=cache_images, use_images=use_images, dev=use_dev, test_dev=test_dev,
                            keep_original_text=keep_original_text, keep_original_image=keep_original_image,
                            keep_processed_image=keep_processed_image, keep_torchvision_image=keep_torchvision_image,
                            train_text_transform=train_text_transform, train_image_transform=train_image_transform,
                            train_torchvision_image_transform=train_torchvision_image_transform,
                            test_torchvision_image_transform=test_torchvision_image_transform,
                            train_mixup_config=train_mixup_config, test_mixup_config=test_mixup_config,
                            train_torchvision_pre_image_transform=train_torchvision_pre_image_transform,
                            test_torchvision_pre_image_transform=test_torchvision_pre_image_transform,
                            test_text_transform=test_text_transform, test_image_transform=test_image_transform,
                            data_dir=data_dir))
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


def create_collage(width, height, images, filled_position=None):
    cols = 2
    rows = 2
    assert len(images) <= cols * rows
    thumbnail_width = width//cols
    thumbnail_height = height//rows
    size = thumbnail_width, thumbnail_height
    new_im = Image.new('RGB', (width, height))
    ims = []
    for im in images:
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            if filled_position is not None and (row, col) not in filled_position:
                continue
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0
    return new_im


def identity(x): return x

class TextImageDataset(Dataset):
    def __init__(self, identifiers: List, texts: List[str], image_locations: List[str], labels: torch.Tensor = None,
                 sample_weights: List[float] = None, cached_images: Dict = None,
                 text_transform=None, image_transform=None, cache_images: bool = True, use_images: bool = True,
                 torchvision_image_transform=None, torchvision_pre_image_transform=None,
                 mixup_config=None,
                 keep_original_text: bool = False, keep_original_image: bool = False,
                 keep_processed_image: bool = False, keep_torchvision_image: bool = False):
        self.texts = list(texts)
        self.identifiers = list(identifiers)
        self.image_locations = image_locations
        from tqdm.auto import tqdm as tqdm, trange
        self.images = dict()
        if use_images:
            if cached_images is not None:
                self.images = cached_images
            elif cache_images:
                self.images = {l: Image.open(l).convert('RGB') for l in tqdm(list(set(image_locations)), "Caching Images in Dataset")}
        self.labels = labels if labels is not None else ([0] * len(texts))
        self.text_transform = text_transform if text_transform is not None else identity
        self.image_transform = image_transform if image_transform is not None else identity
        self.use_images = use_images
        self.sample_weights = [1.0] * len(texts) if sample_weights is None else sample_weights
        assert len(self.sample_weights) == len(self.image_locations) == len(self.texts)
        self.keep_original_text = keep_original_text
        self.keep_original_image = keep_original_image
        self.to_torchvision = get_image2torchvision_transforms()
        self.torchvision_image_transform = torchvision_image_transform if torchvision_image_transform is not None else identity
        self.torchvision_pre_image_transform = torchvision_pre_image_transform if torchvision_pre_image_transform is not None else identity
        self.keep_processed_image = keep_processed_image
        self.keep_torchvision_image = keep_torchvision_image
        self.mixup_config = mixup_config

    def item_getter(self, item):
        text = self.texts[item]
        identifier = self.identifiers[item]
        label = self.labels[item] if self.labels is not None else 0
        sample_weight = self.sample_weights[item]
        # clean_text
        s = Sample({"id": identifier, "text": text, "label": label, "sample_weight": sample_weight, "image": None})

        if self.use_images and (self.keep_torchvision_image or self.keep_original_image or self.keep_processed_image):
            l = self.image_locations[item]
            image = self.images.get(l)
            if image is None:
                image = Image.open(l).convert('RGB')
            s.image = image

        return s

    def process_example(self, sample):
        identifier, text, image, label, sample_weight, mixup = sample["id"], sample["text"], sample["image"], sample["label"], sample["sample_weight"], sample["mixup"]
        # clean_text
        orig_text = text
        text = self.text_transform(text) # Give ID here to retrieve DAB examples
        s = Sample({"id": identifier, "text": text, "label": label, "sample_weight": sample_weight, "mixup": mixup})
        if image is not None:
            if self.keep_original_image:
                s.original_image = image
            if self.keep_processed_image:
                image = self.image_transform(image.copy())
                s.image = image
            if self.keep_torchvision_image:
                torchvision_image = self.torchvision_image_transform(self.to_torchvision(self.torchvision_pre_image_transform(image)))
                s.torchvision_image = torchvision_image

        if self.keep_original_text:
            s.original_text = orig_text

        return s

    def mixup(self, sample):
        indices = random.sample(range(len(self.texts)), random.randint(1, 3))
        samples = [self.item_getter(i) for i in indices] + [sample]
        random.shuffle(samples)
        image = None
        if sample.image is not None:
            positions = random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], len(samples))
            image = create_collage(640, 640, [s.image for s in samples], filled_position=positions)
        text = " ".join([s.text for s in samples])
        label = min(sum([s.label for s in samples]), 1)
        sample_weight = sum([s.sample_weight for s in samples]) / len(indices)
        sample = Sample({"id": -1, "text": text, "label": label, "sample_weight": sample_weight, "image": image})
        return sample

    def __getitem__(self, item):
        sample = self.item_getter(item)
        sample.mixup = False
        if self.mixup_config is not None:
            proba = self.mixup_config.pop("proba", 1.0)
            if random.random() < proba:
                sample = self.mixup(sample)
                sample.mixup = True

        return self.process_example(sample)

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

    def show_mixup(self, item):
        image = self.show(item)
        sample = self.item_getter(item)
        sample.image = image
        sample = self.mixup(sample)
        text = sample.text
        label = sample.label
        image = sample.image
        print("Example Mixup", "\n", text, "|", "Label = ", label)
        image.show()
        return image



class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, images: Union[str, List[str]], image_extensions=(".jpg", ".png", ".jpeg"),
                 cache_images: bool = False, shuffle: bool = False, image_transform=get_image2torchvision_transforms()):
        import os
        if type(images) == str:
            self.images = list(filter(lambda i: any([ex in i for ex in image_extensions]), os.listdir(images)))
            self.images = list(map(lambda i: os.path.join(images, i), self.images))
        else:
            self.images = images

        if shuffle:
            self.images = np.random.permutation(self.images)
        else:
            self.images = sorted(self.images)

        self.image_transform = image_transform
        if cache_images:
            self.images = {i: Image.open(l).convert('RGB') for i, l in enumerate(self.images)} if cache_images else dict()
        self.cache_images = cache_images

    def __getitem__(self, item):
        if self.cache_images:
            return self.image_transform(self.images[item])
        else:
            return self.image_transform(Image.open(self.images[item]).convert('RGB'))

    def __len__(self):
        return len(self.images)

    @classmethod
    def from_images(cls, images, image_transform=get_image2torchvision_transforms()):
        new_inst = ImageFolderDataset([], cache_images=False)
        assert type(images) == dict
        new_inst.images = images
        new_inst.cache_images = True
        return new_inst


class ZipDatasets(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset]):
        assert len(set(list(map(len, datasets)))) == 1  # All datasets are of same length
        assert len(datasets) > 1
        self.datasets = datasets

    def __getitem__(self, item):
        items = [d[item] for d in self.datasets]
        return items

    def __len__(self):
        return len(self.datasets[0])


class NegativeSamplingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, negative_proportion=5):
        self.pos_proba = 1 / (negative_proportion + 1)
        self.negative_proportion = negative_proportion
        self.dataset = dataset

    def __getitem__(self, item):
        if random.random() <= self.pos_proba:
            return [self.dataset[item], self.dataset[item], 1]
        else:
            return [self.dataset[item], self.dataset[random.randint(0, len(self.dataset) - 1)], 0]

    def __len__(self):
        return len(self.dataset)
