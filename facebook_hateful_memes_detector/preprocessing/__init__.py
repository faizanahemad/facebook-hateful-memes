from collections import defaultdict

import torch
import re
from typing import List, Dict, Union, Callable

import contractions
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from albumentations import augmentations as alb
from torch.utils.data import Dataset
from torchvision import transforms
import random

from ..utils import read_json_lines_into_df
from ..utils.sample import Sample


def identity(x): return x

def return_first_arg(x, **kwargs): return x


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
            arr[x_third:x_2third, :] = mean
        if choice == 6:
            arr[:, y_third:y_2third] = mean
        if choice == 7:
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


def tokenize(text):
    text = re.sub(r'(?<=[.,;!])(?=[^\s0-9])', ' ', text)
    text = re.sub('[ ]+', ' ', text)
    return re.split(r"[ ,]+", text)


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
import gensim.downloader as api
from nltk import sent_tokenize
from gensim.models.fasttext import load_facebook_model
from nltk.corpus import stopwords


def isnumber(text):
    try:
        text = int(text)
        return True
    except:
        try:
            t = float(text)
            return True
        except:
            pass
    return False

class TextAugment:
    def __init__(self, count_proba: List[float], choice_probas: Dict[str, float],
                 fasttext_file: str = None, idf_file: str = None, dab_file: str = None):
        self.count_proba = count_proba
        assert 1 - 1e-6 <= sum(count_proba) <= 1 + 1e-6
        assert len(count_proba) >= 1
        assert (len(count_proba) - 1) < sum([v > 0 for v in choice_probas.values()])

        import nlpaug.augmenter.char as nac
        import nlpaug.augmenter.word as naw
        from gensim.similarities.index import AnnoyIndexer

        def first_part_select(text, sp=0.7, lp=0.9):
            splits = text.split()
            if len(splits) <= 2:
                return text
            actual_len = random.randint(int(len(splits) * sp), int(len(splits) * lp))
            if actual_len == len(splits):
                return text
            start = 0
            return " ".join(splits[start:start + actual_len])

        def change_number(text: str, mod_proba=0.9):
            try:
                text = int(text)

            except:
                try:
                    t = float(text)
                    pre, post = text.split(".")
                    pre = int(pre)
                    lp = len(post)
                    if lp == 0:
                        t = int(pre)
                    text = t
                except:
                    pass

            if isinstance(text, (float, int)):
                if random.random() <= mod_proba:
                    rand = random.random()
                    if rand <= 0.15:
                        # Entirely Random
                        text = str("%.2f" % (random.randint(0, 100) + random.random()))
                    elif rand <= 0.4:
                        # Wild change
                        if isinstance(text, int):
                            change_range = max(1, int(0.25 * text))
                            text = text + random.randint(-change_range, change_range)
                        else:
                            change_range = max(1, int(0.25 * pre))
                            pre = pre + random.randint(-change_range, change_range)
                            post = ("%.2f" % random.random()).split(".")[1]
                            text = str(pre) + "." + post
                    elif rand <= 0.7:
                        # Small change
                        if isinstance(text, int):
                            change_range = max(1, int(0.1 * text))
                            text = min(0, text + random.randint(-change_range, change_range))
                        else:
                            change_range = max(1, int(0.1 * pre))
                            pre = min(0, pre + random.randint(-change_range, change_range))
                            post = ("%.2f" % random.random()).split(".")[1]
                            text = str(pre) + "." + post
                    elif rand <= 0.9:
                        # Small change
                        if isinstance(text, int):
                            change_range = max(1, int(0.05 * text))
                            text = min(0, text + random.randint(-change_range, change_range))
                        else:
                            change_range = max(1, int(0.05 * pre))
                            pre = min(0, pre + random.randint(-change_range, change_range))
                            post = post[:-1] + str(random.sample([int(post[-1]) + 1, int(post[-1]) - 1], 1)[0])
                            text = str(pre) + "." + post
                    else:
                        if isinstance(text, int):
                            text = text + random.sample([-1, 1], 1)[0]
                        else:
                            post = post[:-1] + str(random.sample([int(post[-1]) + 1, int(post[-1]) - 1], 1)[0])
                            text = str(pre) + "." + post
            return str(text)

        def number_modify(text):
            splits = list(map(change_number, text.split()))
            return " ".join(splits)

        def gibberish_insert(text):
            words = text.split()
            idx = random.randint(0, len(words) - 1)
            sw = "".join(random.sample("abcdefghijklmnopqrstuvwxyz.!?\",'", random.randint(3, 15)))
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

        punctuation_list = ".,\"'?!@$"

        def punctuation_insert(text):
            words = text.split()
            idx = random.randint(0, len(words) - 1)
            sw = random.sample(punctuation_list, 1)[0]
            words = words[:idx] + ([sw] * random.randint(1, 3)) + words[idx:]
            return " ".join(words)

        def punctuation_continue(text):
            chars = list(text)
            new_text = []
            for i, c in enumerate(chars):
                if i == 0 or i == len(chars) - 1:
                    new_text.append(c)
                    continue
                if c in punctuation_list and not chars[i-1].isnumeric() and not chars[i+1].isnumeric():
                    if random.random() < 0.5:
                        puncts = "".join([random.sample(".,\"'?!", 1)[0]] * random.randint(1, 3))
                    else:
                        puncts = "".join([c] * random.randint(1, 3))

                    if random.random() < 0.5:
                        new_text.append(c)
                        new_text.append(puncts)
                    else:
                        new_text.append(puncts)
                        new_text.append(c)
                else:
                    new_text.append(c)
            return "".join(new_text)

        def punctuation_replace(text):
            chars = list(text)
            for i, c in enumerate(chars):
                if i == 0 or i == len(chars) - 1:
                    continue
                if c in punctuation_list and not chars[i-1].isnumeric() and not chars[i+1].isnumeric():
                    chars[i] = random.sample(punctuation_list, 1)[0]
            return "".join(chars)

        def punctuation_strip(text):
            chars = list(text)
            for i, c in enumerate(chars):
                if i == 0 or i == len(chars) - 1:
                    continue
                if c in punctuation_list and not chars[i-1].isnumeric() and not chars[i+1].isnumeric():
                    if random.random() < 0.5:
                        chars[i] = " "
            text = "".join(chars)
            text = " ".join([w.strip() for w in text.split()])
            return text

        def word_join(text):
            words = text.split()
            probas = [1 / (np.sqrt(len(w)) + np.sqrt(len(words[i+1]))) if not isnumber(w) and not isnumber(words[i+1]) else 0.0 for i, w in enumerate(words[:-1])]
            if len(words) <= 2:
                return text
            idx = random.choices(range(len(words) - 1), probas)[0]
            w1 = words[idx] + words[idx + 1]
            words = words[:idx] + [w1] + words[idx + 2:]
            return " ".join(words)

        self.augs = ["keyboard", "ocr", "char_insert", "char_substitute", "char_swap", "char_delete",
                     "word_insert", "word_substitute", "w2v_insert", "w2v_substitute", "text_rotate",
                     "stopword_insert", "word_join", "word_cutout", "first_part_select", "number_modify",
                     "fasttext", "glove_twitter", "glove_wiki", "word2vec", "gibberish_insert",
                     "synonym", "split", "sentence_shuffle", "one_third_cut", "half_cut", "part_select",
                     "punctuation_insert", "punctuation_replace", "punctuation_strip", "punctuation_continue", "dab"]
        assert len(set(list(choice_probas.keys())) - set(self.augs)) == 0
        self.augments = dict()
        self.indexes = dict()
        if not choice_probas.keys().isdisjoint(["glove_wiki", "glove_twitter", "word2vec", "fasttext", "word_cutout"]):
            assert idf_file is not None
            tfidf = pd.read_csv(idf_file)
            tfidf['token'] = tfidf['token'].apply(lambda x: x.lower() if isinstance(x, str) else x)
            self.idfs = dict(zip(tfidf.to_dict()['token'].values(), tfidf.to_dict()['idf'].values()))
            self.max_idf_score = tfidf.idf.max()
            tfidf = tfidf[tfidf["idf"] < tfidf["idf"].max()]
            tfidf["kw"] = np.log1p(tfidf["frequency"] * tfidf["idf"])
            max_kw = tfidf["kw"].max()
            zd = (max_kw - tfidf["kw"]).mean()
            tfidf["kw"] = (max_kw - tfidf["kw"]) / zd
            self.kw_select_proba = dict(zip(tfidf.to_dict()['token'].values(), tfidf.to_dict()['kw'].values()))

        for k, v in choice_probas.items():
            if v <= 0:
                continue
            if k == "number_modify":
                self.augments["number_modify"] = number_modify
            if k == "first_part_select":
                self.augments["first_part_select"] = first_part_select
            if k == "part_select":
                self.augments["part_select"] = part_select
            if k == "punctuation_insert":
                self.augments["punctuation_insert"] = punctuation_insert
            if k == "punctuation_continue":
                self.augments["punctuation_continue"] = punctuation_continue
            if k == "punctuation_replace":
                self.augments["punctuation_replace"] = punctuation_replace
            if k == "punctuation_strip":
                self.augments["punctuation_strip"] = punctuation_strip
            if k == "gibberish_insert":
                self.augments["gibberish_insert"] = gibberish_insert
            if k == "stopword_insert":
                self.augments["stopword_insert"] = stopword_insert
            if k == "word_join":
                self.augments["word_join"] = word_join
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

            if k == "dab":
                assert isinstance(dab_file, str)
                dab = pd.read_csv(dab_file).values
                dab_store = defaultdict(list)
                for d in dab:
                    dab_store[int(d[0])].append(d[1])
                self.dab_store = dab_store

            if k == "fasttext":
                assert fasttext_file is not None
                self.augments["fasttext"] = load_facebook_model(fasttext_file)
                self.indexes["fasttext"] = AnnoyIndexer(self.augments["fasttext"], 32)
            if k == "word2vec":
                self.augments["word2vec"] = api.load("word2vec-google-news-300")
                self.indexes["word2vec"] = AnnoyIndexer(self.augments["word2vec"], 32)
            if k == "glove_twitter":
                self.augments["glove_twitter"] = api.load("glove-twitter-100")
                self.indexes["glove_twitter"] = AnnoyIndexer(self.augments["glove_twitter"], 32)
            if k == "glove_wiki":
                self.augments["glove_wiki"] = api.load("glove-wiki-gigaword-100")
                self.indexes["glove_wiki"] = AnnoyIndexer(self.augments["glove_wiki"], 32)

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

    def idf_proba(self, text):
        idfs: Dict[str, float] = self.idfs
        max_score = self.max_idf_score
        words = text.lower().split()
        idf_scores = [(w, idfs[w] if w in idfs and not isnumber(w) else max_score) for w in words]
        max_score = max(list([sc[1] for sc in idf_scores]))
        max_minus_score = [max_score - s for w, s in idf_scores]
        z = sum(max_minus_score) / len(words)
        if z == 0:
            word_scores = [1 / len(words)] * len(words)
        else:
            p = 0.6
            word_scores = [min(p * s / z, 1) for s in max_minus_score]
        return list(zip(words, word_scores))

    def word_cutout(self, text):
        words = text.split()
        proba = self.idf_proba(text)
        # np.mean(list(proba.values()))
        probas = [(1 / np.sqrt(len(w))) * p for w, (ws, p) in zip(words, proba)]
        if len(words) <= 3:
            return text

        cut_idx = random.choices(range(len(words)), probas)[0]
        words = words[:cut_idx] + words[cut_idx + 1:]
        return " ".join(words)

    def __fasttext_replace__(self, tm, indexer, text):
        tokens = text.split()
        t_2_i = {w: i for i, w in enumerate(tokens)}
        if len(tokens) <= 3:
            return text
        proba = self.idf_proba(text)
        sampled = random.choices(list(proba.keys()), list(proba.values()), k=1)[0]
        sampled_idx = t_2_i[sampled]
        # candidates = [w for d, w in self.augments["fasttext"].get_nearest_neighbors(sampled, 10)]
        candidates = [w for w, d in tm.most_similar(sampled, topn=10, indexer=indexer)][1:5]
        replacement = random.sample(candidates, k=1)[0]
        tokens[sampled_idx] = replacement
        return " ".join(tokens)

    def __w2v_replace__(self, tm, indexer, text):
        # TODO: Test if makng length aware in proba selection performs better
        tokens = text.split()
        t_2_i = {w: i for i, w in enumerate(tokens)}
        if len(t_2_i) <= 3:
            return text
        words, proba = zip(*self.idf_proba(text))
        success = False
        repeat_count = 0
        while not success and repeat_count <= 10:
            repeat_count += 1
            sampled = random.choices(words, proba, k=1)[0]
            if sampled in tm:
                candidates = [w for w, d in tm.most_similar(sampled, topn=10, indexer=indexer)][1:]
                success = True
        if not success:
            return text
        sampled_idx = t_2_i[sampled]
        replacement = random.sample(candidates, k=1)[0]
        tokens[sampled_idx] = replacement
        return " ".join(tokens)

    def __call__(self, text, **kwargs):
        original_text = text
        count = np.random.choice(list(range(len(self.count_proba))), 1, replace=False, p=self.count_proba)[0]
        augs = np.random.choice(self.augs, count, replace=False, p=self.choice_probas)
        for aug in augs:
            if len(text.split()) < 2:
                break
            try:
                if aug == "fasttext":
                    text = self.__fasttext_replace__(self.augments[aug], self.indexes[aug], text)
                elif aug == "word_cutout":
                    text = self.word_cutout(text)
                elif aug == "dab":
                    identifier = int(kwargs["identifier"])
                    text = random.sample(self.dab_store[identifier], 1)[0]
                elif aug in ["glove_twitter", "glove_wiki", "word2vec"]:
                    text = self.__w2v_replace__(self.augments[aug], self.indexes[aug], text)
                elif callable(self.augments[aug]):
                    text = self.augments[aug](text)
                elif hasattr(self.augments[aug], "augment"):
                    text = self.augments[aug].augment(text)
                else:
                    raise ValueError()
            except Exception as e:
                print("Exception for: ", aug, "|", "Original Text", original_text, "Final Text", text, "|", augs, e)
        return text


def get_image2torchvision_transforms():
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


def get_transforms_for_bbox_methods():
    def get_alb(aug):
        def augment(image):
            return Image.fromarray(aug(image=np.array(image, dtype=np.uint8))['image'])
        return augment

    transforms_for_bbox_methods = transforms.RandomChoice([DefinedRotation(90), DefinedRotation(15), HalfSwap(), QuadrantCut(),
                                                           DefinedAffine(0, scale=(0.6, 0.6)), DefinedAffine(0, translate=(0.25, 0.25)),
                                                           DefinedAffine(0, translate=(0.1, 0.1)), transforms.RandomAffine(0, scale=(0.5, 0.5)),
                                                           transforms.RandomAffine(0, scale=(0.75, 0.75)), transforms.RandomAffine(0, scale=(1.25, 1.25)),
                                                           transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]),
                                                           transforms.Compose([transforms.Resize(480), transforms.CenterCrop(400)]),
                                                           transforms.Grayscale(num_output_channels=3),
                                                           transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), identity,
                                                           get_alb(alb.transforms.GridDropout(ratio=0.15, holes_number_x=10, holes_number_y=10,
                                                                                              random_offset=False, p=1.0)),
                                                           get_alb(alb.transforms.GridDropout(ratio=0.25, holes_number_x=16, holes_number_y=16,
                                                                                              random_offset=False, p=1.0)),
                                                           get_alb(alb.transforms.GridDropout(ratio=0.35, holes_number_x=32, holes_number_y=32,
                                                                                              random_offset=False, p=1.0)),


                                                           ])
    return transforms_for_bbox_methods


def get_image_transforms_pytorch(mode="easy"):

    def get_imgaug(aug):
        def augment(image):
            return Image.fromarray(aug(image=np.array(image, dtype=np.uint8)))
        return augment

    def get_alb(aug):
        def augment(image):
            return Image.fromarray(aug(image=np.array(image, dtype=np.uint8))['image'])
        return augment

    p = 0.1
    param1 = 0.05
    rotation = 15
    cutout_max_count = 1
    cutout_size = 0.1
    grid_random_offset = False
    distortion_scale = 0.1
    grid_ratio = 0.0
    cutout_proba = 0.5
    alb_proba = 0.1
    alb_dropout_proba = 0.75
    color_augs = []
    affine_zoom = 0.25
    affine_translate = 0.15
    if mode == "hard":
        grid_ratio = 0.15
        p = 0.25
        param1 = 0.15
        rotation = 30
        cutout_max_count = 3
        cutout_size = 0.25
        grid_random_offset = True
        distortion_scale = 0.25
        cutout_proba = 1.0
        alb_proba = 0.4
        alb_dropout_proba = 1.0
        element_wise_add = 25
        affine_zoom = 0.5
        affine_translate = 0.2
        color_augs = [transforms.RandomChoice([
            get_imgaug(iaa.AddElementwise((-element_wise_add, element_wise_add), per_channel=0.5)),
            get_imgaug(iaa.imgcorruptlike.MotionBlur(severity=1)),
            get_imgaug(iaa.AllChannelsCLAHE()),
            get_imgaug(iaa.LogContrast(gain=(0.6, 1.4))),
            get_imgaug(iaa.pillike.Autocontrast((10, 20), per_channel=True))
        ])]

    def get_cutout():
        cut = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=cutout_proba, scale=(0.05, cutout_size), ratio=(0.3, 3.3), value='random', inplace=False),
            transforms.ToPILImage(),
        ])
        return cut
    cut = get_cutout()

    def cutout(img):
        for _ in range(random.randint(1, cutout_max_count)):
            img = cut(img)
        return img

    preprocess = transforms.Compose([
        transforms.RandomGrayscale(p=p),
        transforms.RandomHorizontalFlip(p=p),
        transforms.RandomPerspective(distortion_scale=distortion_scale, p=p),
        transforms.ColorJitter(brightness=param1, contrast=param1, saturation=param1, hue=param1),
        transforms.RandomChoice([
            cutout,
            get_alb(alb.transforms.GridDropout(ratio=0.35+grid_ratio, holes_number_x=8, holes_number_y=8, random_offset=grid_random_offset, p=alb_dropout_proba)),
            get_alb(alb.transforms.GridDropout(ratio=0.35+grid_ratio, holes_number_x=16, holes_number_y=16, random_offset=grid_random_offset, p=alb_dropout_proba)),
            get_alb(alb.transforms.GridDropout(ratio=0.35+grid_ratio, holes_number_x=10, holes_number_y=10, random_offset=grid_random_offset, p=alb_dropout_proba)),
            get_alb(alb.transforms.GridDropout(ratio=0.35+grid_ratio, holes_number_x=32, holes_number_y=32, random_offset=grid_random_offset, p=alb_dropout_proba)),
            get_alb(alb.transforms.GridDropout(ratio=0.2+grid_ratio, holes_number_x=8, holes_number_y=8, random_offset=grid_random_offset, p=alb_dropout_proba)),
            get_alb(alb.transforms.GridDropout(ratio=0.2+grid_ratio, holes_number_x=16, holes_number_y=16, random_offset=grid_random_offset, p=alb_dropout_proba)),
            get_alb(alb.transforms.GridDropout(ratio=0.2+grid_ratio, holes_number_x=10, holes_number_y=10, random_offset=grid_random_offset, p=alb_dropout_proba)),
            get_alb(alb.transforms.GridDropout(ratio=0.2+grid_ratio, holes_number_x=32, holes_number_y=32, random_offset=grid_random_offset, p=alb_dropout_proba)),
            get_alb(alb.transforms.CoarseDropout(max_holes=32, max_height=64, max_width=64, min_holes=8, min_height=32, min_width=32, fill_value=0, p=alb_dropout_proba)),
            get_alb(alb.transforms.CoarseDropout(max_holes=16, max_height=128, max_width=128, min_holes=4, min_height=64, min_width=64, fill_value=0, p=alb_dropout_proba)),
        ]),
        transforms.RandomOrder([
             get_alb(alb.transforms.MedianBlur(p=alb_proba)),
             get_alb(alb.transforms.RandomGamma(p=alb_proba)),
             get_alb(alb.transforms.RGBShift(p=alb_proba)),
             get_alb(alb.transforms.MotionBlur(p=alb_proba)),
             get_alb(alb.transforms.ImageCompression(95,p=alb_proba)),
             get_alb(alb.transforms.Equalize(p=alb_proba)),
             get_alb(alb.transforms.Posterize(num_bits=4, always_apply=False, p=alb_proba)),
             get_alb(alb.transforms.Solarize(threshold=128, always_apply=False, p=alb_proba)),
             get_alb(alb.transforms.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=alb_proba)),
        ]),
        transforms.RandomChoice([
            transforms.RandomRotation(rotation),
            transforms.RandomVerticalFlip(p=1.0),
            DefinedRotation(90),
            transforms.RandomAffine(
                0,
                translate=(affine_translate, affine_translate),
                scale=(1 - affine_zoom, 1 + affine_zoom),  # 0.6 -> Zoom out, 1.4 -> Zoom in
                shear=25,
            ),
            transforms.RandomResizedCrop(640, scale=(0.6, 0.8)),  # Zoom in
            transforms.RandomResizedCrop(360, scale=(0.4, 0.8)),
        ]),
    ] + color_augs)
    return preprocess


def get_csv_datasets(train_file, test_file, image_dir, numeric_file, numeric_file_dim,
                     embed1, embed2, embed1_dim, embed2_dim,
                     image_extension=".png",
                     numeric_regularizer: Callable=None,
                     train_text_transform=None, train_image_transform=None,
                     train_torchvision_pre_image_transform=None, test_torchvision_pre_image_transform=None,
                     test_text_transform=None, test_image_transform=None,
                     cache_images: bool = True, use_images: bool = True, dev: bool = False, test_dev: bool = True,
                     keep_original_text: bool = False, keep_original_image: bool = False,
                     train_mixup_config=None, test_mixup_config=None,
                     keep_processed_image: bool = False, keep_torchvision_image: bool = False):
    use_dev = dev
    joiner_p = lambda img: (os.path.join(image_dir, img)) if (img is not None and type(img) == str and img != "nan") else None
    from torch.utils.data import Subset
    def joiner(img):
        img = joiner_p(img)
        if img is None:
            return img
        if img.endswith(image_extension):
            return img
        return img + image_extension
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    perm = np.random.permutation(len(train))
    train = train.iloc[perm]
    sp = int(0.1 * len(train))
    dev = train[:sp]
    assert (numeric_file is None and numeric_file_dim is None) or (numeric_file is not None and numeric_file_dim is not None)
    assert (embed1 is None and embed1_dim is None) or (embed1 is not None and embed1_dim is not None)
    assert (embed2 is None and embed2_dim is None) or (embed2 is not None and embed2_dim is not None)
    numeric_train = None
    numeric_test = None
    numeric_dev = None
    if numeric_file is not None:
        assert numeric_file_dim[0] == train.shape[0] + test.shape[0]
        numeric_file = np.memmap(numeric_file, dtype='float32', mode='r', shape=numeric_file_dim)
        numeric_train = Subset(numeric_file, list(range(train.shape[0])))
        numeric_test = Subset(embed1, list(range(train.shape[0], train.shape[0] + test.shape[0])))
        numeric_train = Subset(numeric_train, perm)
        numeric_dev = Subset(numeric_train, list(range(sp)))
        if test_dev:
            numeric_train = Subset(numeric_train, list(range(sp, len(numeric_train))))

    embed1_train = None
    embed1_test = None
    embed1_dev = None
    if embed1 is not None:
        assert embed1_dim[0] == train.shape[0] + test.shape[0]
        embed1 = np.memmap(embed1, dtype='float32', mode='r', shape=embed1_dim)
        embed1_train = Subset(embed1, list(range(train.shape[0])))
        embed1_test = Subset(embed1, list(range(train.shape[0], train.shape[0] + test.shape[0])))
        embed1_train = Subset(embed1_train, perm)
        embed1_dev = Subset(embed1_train, list(range(sp)))
        if test_dev:
            embed1_train = Subset(embed1_train, list(range(sp, len(embed1_train))))

    embed2_train = None
    embed2_test = None
    embed2_dev = None
    if embed2 is not None:
        assert embed2_dim[0] == train.shape[0] + test.shape[0]
        embed2 = np.memmap(embed2, dtype='float32', mode='r', shape=embed2_dim)
        embed2_train = Subset(embed2, list(range(train.shape[0])))
        embed2_test = Subset(embed2, list(range(train.shape[0], train.shape[0] + test.shape[0])))
        embed2_train = Subset(embed2_train, perm)
        embed2_dev = Subset(embed2_train, list(range(sp)))
        if test_dev:
            embed2_train = Subset(embed2_train, list(range(sp, len(embed2_train))))

    if test_dev:
        train = train[sp:]

    dev["img"] = list(map(joiner, dev.img))
    train["img"] = list(map(joiner, train.img))
    test["img"] = list(map(joiner, test.img))

    rd = dict(train=train, test=test, dev=dev,
              numeric_train=numeric_train, numeric_test=numeric_test, numeric_dev=numeric_dev,
              embed1_train=embed1_train, embed1_test=embed1_test, embed1_dev=embed1_dev,
              embed2_train=embed2_train, embed2_test=embed2_test, embed2_dev=embed2_dev,
              metadata=dict(cache_images=cache_images, use_images=use_images, dev=use_dev, numeric_regularizer=numeric_regularizer,
                            keep_original_text=keep_original_text, keep_original_image=keep_original_image,
                            keep_processed_image=keep_processed_image, keep_torchvision_image=keep_torchvision_image,
                            train_text_transform=train_text_transform, train_image_transform=train_image_transform,
                            train_torchvision_pre_image_transform=train_torchvision_pre_image_transform,
                            train_mixup_config=train_mixup_config, test_mixup_config=test_mixup_config,
                            test_torchvision_pre_image_transform=test_torchvision_pre_image_transform,
                            test_text_transform=test_text_transform, test_image_transform=test_image_transform))
    return rd


def get_datasets(data_dir, train_text_transform=None, train_image_transform=None,
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
              numeric_train=None, numeric_dev=None, numeric_test=None,
              embed1_train=None, embed1_test=None, embed1_dev=None,
              embed2_train=None, embed2_test=None, embed2_dev=None,
              submission_format=submission_format,
              metadata=dict(cache_images=cache_images, use_images=use_images, dev=use_dev, test_dev=test_dev,
                            keep_original_text=keep_original_text, keep_original_image=keep_original_image,
                            keep_processed_image=keep_processed_image, keep_torchvision_image=keep_torchvision_image,
                            train_text_transform=train_text_transform, train_image_transform=train_image_transform,
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


def make_weights_for_uda(labels, weight_per_class: Dict = None):
    labels = labels.numpy()
    from collections import Counter
    count = Counter(labels)
    N = len(labels)
    last_label = max(labels)
    last_label_occ = count[last_label]
    last_excluded_count = N - last_label_occ
    last_excluded_ratio = last_excluded_count / N
    if weight_per_class is None:
        weight_per_class = {clas: last_excluded_count / float(occ) for clas, occ in count.items() if clas != last_label}
        tot_weights = sum(list(weight_per_class.values()))
        weight_per_class = {clas: (w / tot_weights) * last_excluded_ratio for clas, w in weight_per_class.items()}
        weight_per_class[last_label] = 1.0 - last_excluded_ratio
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


class TextImageDataset(Dataset):
    def __init__(self, identifiers: List, texts: List[str], image_locations: List[str], labels: torch.Tensor = None,
                 numbers: np.ndarray = None, embed1: np.ndarray = None, embed2: np.ndarray = None,
                 sample_weights: List[float] = None, cached_images: Dict = None,
                 text_transform=None, image_transform=None, cache_images: bool = True, use_images: bool = True,
                 torchvision_pre_image_transform=identity, numeric_regularizer: Callable = identity,
                 mixup_config=None,
                 keep_original_text: bool = False, keep_original_image: bool = False,
                 keep_processed_image: bool = False, keep_torchvision_image: bool = False):
        self.texts = list(texts)
        self.identifiers = list(identifiers)
        self.image_locations = image_locations
        if numbers is not None:
            assert isinstance(numbers, np.ndarray)
            self.numbers = numbers
        if embed1 is not None:
            assert isinstance(embed1, np.ndarray)
            self.embed1 = embed1
        if embed2 is not None:
            assert isinstance(embed2, np.ndarray)
            self.embed2 = embed2
        self.numeric_regularizer = numeric_regularizer if numeric_regularizer is not None else identity
        from tqdm.auto import tqdm as tqdm
        self.images = dict()
        if use_images:
            if cached_images is not None:
                self.images = cached_images
            elif cache_images:
                self.images = {l: Image.open(l).convert('RGB') if l is not None else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for l in tqdm(list(set(image_locations)), "Caching Images in Dataset")}
        self.labels = labels if labels is not None else ([0] * len(texts))
        self.text_transform = text_transform if text_transform is not None else return_first_arg
        self.image_transform = image_transform if image_transform is not None else identity
        self.use_images = use_images
        self.sample_weights = [1.0] * len(texts) if sample_weights is None else sample_weights
        assert len(self.sample_weights) == len(self.image_locations) == len(self.texts)
        self.keep_original_text = keep_original_text
        self.keep_original_image = keep_original_image
        self.to_torchvision = get_image2torchvision_transforms()
        self.torchvision_pre_image_transform = torchvision_pre_image_transform if torchvision_pre_image_transform is not None else identity
        self.keep_processed_image = keep_processed_image
        self.keep_torchvision_image = keep_torchvision_image
        self.mixup_config = mixup_config

    def item_getter(self, item):
        text = self.texts[item]
        identifier = self.identifiers[item]
        label = self.labels[item] if self.labels is not None else 0
        sample_weight = self.sample_weights[item]
        s = Sample({"id": identifier, "text": text, "label": label, "sample_weight": sample_weight, "image": None})
        if hasattr(self, "numbers"):
            s.numbers = torch.tensor(self.numeric_regularizer(self.numbers[item]))
        if hasattr(self, "embed1"):
            s.embed1 = torch.tensor(self.embed1[item])
        if hasattr(self, "embed2"):
            s.embed2 = torch.tensor(self.embed2[item])

        if self.use_images and (self.keep_torchvision_image or self.keep_original_image or self.keep_processed_image):
            l = self.image_locations[item]
            image = self.images.get(l)
            if image is None:
                image = Image.open(l).convert('RGB') if l is not None else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            s.image = image

        return s

    def process_example(self, sample):
        s = Sample(sample)
        # clean_text
        orig_text = s["text"]
        text = self.text_transform(orig_text, identifier=s.id)  # Give ID here to retrieve DAB examples
        s.text = text
        image = s["image"]
        if image is not None:
            if self.keep_original_image:
                s.original_image = image
            if self.keep_processed_image:
                image = self.image_transform(image.copy())
                s.image = image
            if self.keep_torchvision_image:
                torchvision_image = self.to_torchvision(self.torchvision_pre_image_transform(image))
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
        sample_weight = sum([s.sample_weight for s in samples]) / len(samples)
        sample = Sample({"id": -1, "text": text, "label": label, "sample_weight": sample_weight, "image": image})
        if hasattr(self, "numbers"):
            sample.numbers = torch.stack([s.numbers for s in samples]).mean(0)
        if hasattr(self, "embed1"):
            sample.embed1 = torch.stack([s.embed1 for s in samples]).mean(0)
        if hasattr(self, "embed2"):
            sample.embed2 = torch.stack([s.embed2 for s in samples]).mean(0)

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
            from tqdm.auto import tqdm as tqdm
            self.images = {i: Image.open(l).convert('RGB') for i, l in tqdm(list(enumerate(self.images)))} if cache_images else dict()
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


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = list(texts)

    def __getitem__(self, item):
        return self.texts[item]

    def __len__(self):
        return len(self.texts)


