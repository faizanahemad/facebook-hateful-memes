import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import re
import contractions


def init_weight(param, initializer, nonlinearity, nonlinearity_param=None):
    initializer = getattr(nn.init, initializer)
    initializer(param, nn.init.calculate_gain(nonlinearity, nonlinearity_param))


def init_bias(param):
    nn.init.normal_(param, 0, 0.001)


def init_fc(layer, initializer, nonlinearity, nonlinearity_param=None):
    init_weight(layer.weight, initializer, nonlinearity, nonlinearity_param)
    try:
        init_bias(layer.bias)
    except AttributeError:
        pass


def clean_text(text):
    EMPTY = ' '
    assert text is not None
    assert type(text) == str
    text = text.lower()
    text = contractions.fix(text)
    text = text.replace("'", " ").replace('"', " ")
    text = text.replace("\n", " ").replace("(", " ").replace(")", " ").replace("\r", " ").replace("\t", " ").lower()
    text = re.sub('<pre><code>.*?</code></pre>', EMPTY, text)
    text = re.sub('<code>.*?</code>', EMPTY, text)

    def replace_link(match):
        return EMPTY if re.match('[a-z]+://', match.group(1)) else match.group(1)

    text = re.sub('<a[^>]+>(.*)</a>', replace_link, text)
    text = re.sub('<.*?>', EMPTY, text)
    return text
