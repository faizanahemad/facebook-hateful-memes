import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn

from facebook_hateful_memes_detector import init_fc


class Squeeze(nn.Module):
    def forward(self, input):
        return input.squeeze(1)


class Transpose(nn.Module):
    def forward(self, input):
        return input.transpose(1, 2)


class WordChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels, strides):
        super(WordChannelReducer, self).__init__()
        conv = nn.Conv1d(in_channels, in_channels * strides, 1, 1, padding=0, groups=4, bias=False)
        init_fc(conv, "leaky_relu")
        pool = nn.AvgPool1d(strides)
        conv2 = nn.Conv1d(in_channels * strides, out_channels, 1, 1, padding=0, groups=2, bias=False)
        self.layers = nn.Sequential(Transpose(), conv, nn.LeakyReLU(), pool, conv2, Transpose())


    def forward(self, x):
        return self.layers(x)
