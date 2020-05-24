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
from ..utils import clean_text


class TextImageDataset(Dataset):
    def __init__(self, texts: List[str], image_locations: List[str], labels: torch.Tensor=None,
                 text_transform=None, image_transform=None, cache_images: bool = True):
        self.texts = texts
        self.image_locations = image_locations
        self.images = {l: Image.open(l).convert('RGB') for l in image_locations} if cache_images else dict()
        self.labels = labels
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.cache_images = cache_images

    def __getitem__(self, item):
        l = self.image_locations[item]
        image = self.images.get(l)
        if image is None:
            image = Image.open(l).convert('RGB')
        image = self.image_transform(image.copy()) if self.image_transform is not None else image
        text = self.texts[item]
        text = clean_text(text)
        label = self.labels[item] if self.labels is not None else 0
        text = self.text_transform(text) if self.text_transform is not None else text
        return text, image, label

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


