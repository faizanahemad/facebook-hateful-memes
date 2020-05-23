import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, AlbertTokenizer


class ALBertResnet(nn.Module):
    def __init__(self, add_special_tokens=True, pad_to_max_length=True, max_length=128):
        super(ALBertResnet, self).__init__()

    def compose(self, text, img):
        pass

    def forward(self, text, img):
        pass

    def predict(self, text, img):
        pass


