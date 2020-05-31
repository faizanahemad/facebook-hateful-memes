import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import contractions
import pandas as pd
import jsonlines
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_PADDING_INDEX
from spacy import glossary


def get_all_tags():

    # https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    # https://github.com/nltk/nltk/blob/4e59677df364841c1a23dabfde0317388997aa6d/nltk/sem/relextract.py#L31
    deps = get_universal_deps_indices()
    penn = get_penn_treebank_pos_tag_indices()
    upos = get_pos_tag_indices()
    spacy_glossary = list(glossary.GLOSSARY.keys())

    nltk_ner_tags = ['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION',
            'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE'] + ['LOC', 'PER', 'ORG'] + ['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION',
            'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE', 'FACILITY', 'GPE', 'O']
    snlp_list = ["NUMBER", "ORDINAL", "MONEY", "DATE", "TIME", "CAUSE_OF_DEATH", "CITY",
                 "COUNTRY", "CRIMINAL_CHARGE", "EMAIL", "HANDLE", "IDEOLOGY", "NATIONALITY", "RELIGION", "STATE_OR_PROVINCE", "TITLE", "URL"]
    all_list = deps + penn + upos + spacy_glossary + nltk_ner_tags + snlp_list
    tags = list(set(list(map(lambda x: x.lower(), all_list))))
    return dict(zip(tags, range(1, len(tags)+1)))


def get_universal_deps_indices():
    """
    See `https://spacy.io/api/annotation#dependency-parsing` for the list

    :return:
    """
    tags = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case",
            "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj",
            "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat",
            "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod",
            "obj", "obl", "orphan", "parataxis", "punct", "reparandum",
            "root", "vocative", "xcomp"]
    spacy_deps = ['',
                  'ROOT',
                  'acl',
                  'acomp',
                  'advcl',
                  'advmod',
                  'agent',
                  'amod',
                  'appos',
                  'attr',
                  'aux',
                  'auxpass',
                  'case',
                  'cc',
                  'ccomp',
                  'compound',
                  'conj',
                  'csubj',
                  'csubjpass',
                  'dative',
                  'dep',
                  'det',
                  'dobj',
                  'expl',
                  'intj',
                  'mark',
                  'meta',
                  'neg',
                  'nmod',
                  'npadvmod',
                  'nsubj',
                  'nsubjpass',
                  'nummod',
                  'oprd',
                  'parataxis',
                  'pcomp',
                  'pobj',
                  'poss',
                  'preconj',
                  'predet',
                  'prep',
                  'prt',
                  'punct',
                  'quantmod',
                  'relcl',
                  'xcomp']
    snlp_deps = ['compound:prt', 'nmod:poss', 'tmod', 'pass', 'O']
    tags = tags + spacy_deps + snlp_deps
    tags = list(map(lambda x: x.lower(), tags))
    tags = list(set(tags))
    return tags


def get_penn_treebank_pos_tag_indices():
    """
    See `nltk.help.upenn_tagset()` or `https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html`

    :return: dict of all nltk pos tags from penn tree bank as dict(str->index)
    """
    pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR",
                "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS",
                "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS",
                "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
                "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",
                "$", "''", "(", ")", ",", "--", ".", "::",
                "_SP", "HYPH", 'NFP', ':', 'XX', '-LRB-', '-RRB-', '',
                'ADD', 'AFX']
    spacy_list = ['$',
                  "''",
                  ',',
                  '-LRB-',
                  '-RRB-',
                  '.',
                  ':',
                  'ADD',
                  'AFX',
                  'CC',
                  'CD',
                  'DT',
                  'EX',
                  'FW',
                  'HYPH',
                  'IN',
                  'JJ',
                  'JJR',
                  'JJS',
                  'LS',
                  'MD',
                  'NFP',
                  'NN',
                  'NNP',
                  'NNPS',
                  'NNS',
                  'PDT',
                  'POS',
                  'PRP',
                  'PRP$',
                  'RB',
                  'RBR',
                  'RBS',
                  'RP',
                  'SYM',
                  'TO',
                  'UH',
                  'VB',
                  'VBD',
                  'VBG',
                  'VBN',
                  'VBP',
                  'VBZ',
                  'WDT',
                  'WP',
                  'WP$',
                  'WRB',
                  'XX',
                  '_SP']
    pos_tags = list(set(pos_tags + spacy_list))
    pos_tags = list(map(lambda x: x.lower(), pos_tags))
    return pos_tags


def get_pos_tag_indices():
    pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET",
                "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
    pos_tags = list(map(lambda x: x.lower(), pos_tags))
    return pos_tags


def init_weight(param, initializer, nonlinearity, nonlinearity_param=None):
    initializer = getattr(nn.init, initializer)
    initializer(param, nn.init.calculate_gain(nonlinearity, nonlinearity_param))


def init_bias(param):
    nn.init.normal_(param, 0, 0.001)


def init_fc(layer, nonlinearity, nonlinearity_param=None):
    init_weight(layer.weight, "xavier_uniform_", nonlinearity, nonlinearity_param)
    try:
        init_bias(layer.bias)
    except AttributeError:
        pass


# def init_fc(layer, nonlinearity, nonlinearity_param=None):
#     gain = nn.init.calculate_gain(nonlinearity, nonlinearity_param)
#     for name, param in layer.named_parameters():
#         if 'bias' in name:
#             nn.init.normal_(param, 0.0001)
#         elif 'weight' in name:
#             nn.init.xavier_uniform(param, gain)


def read_json_lines_into_df(file):
    lines = []
    with jsonlines.open(file) as reader:
        for obj in reader:
            lines.append(obj)
    return pd.DataFrame.from_records(lines)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            sigma = self.sigma * 1.0/np.sqrt(x.size(-1))
            scale = sigma * x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True


def pad_tensor(tensor, length, padding_index=DEFAULT_PADDING_INDEX):
    """ Pad a ``tensor`` to ``length`` with ``padding_index``.

    Args:
        tensor (torch.Tensor [n, ...]): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int, optional): Index to pad tensor with.

    Returns
        (torch.Tensor [length, ...]) Padded Tensor.
    """
    n_padding = length - tensor.shape[0]
    if n_padding < 0:
        return tensor[:length]
    if n_padding == 0:
        return tensor
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return torch.cat((tensor, padding), dim=0)


def stack_and_pad_tensors(batch, max_len=None, padding_index=DEFAULT_PADDING_INDEX, dim=0):
    """ Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.
    Modified from ``https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/encoders/text/text_encoder.html``

    Args:
        batch (:class:`list` of :class:`torch.Tensor`): Batch of tensors to pad.
        max_len (int, optional): Length of padding for each sentence.
        padding_index (int, optional): Index to pad tensors with.
        dim (int, optional): Dimension on to which to concatenate the batch of tensors.

    Returns
        BatchedSequences(torch.Tensor, torch.Tensor): Padded tensors and original lengths of
            tensors.
            :param max_len:
    """
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths) if max_len is None else max_len
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = torch.stack(padded, dim=dim).contiguous()
    for _ in range(dim):
        lengths = lengths.unsqueeze(0)
    return padded


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


