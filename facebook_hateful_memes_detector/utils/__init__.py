import operator
import copy
import gc
import math
import operator
import os
import random
import re
import time
from typing import List, Tuple, Dict, Callable
from typing import Optional

import jsonlines
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sample import *
from sklearn.metrics import accuracy_score
from spacy import glossary
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, LayerNorm, TransformerEncoderLayer, CrossEntropyLoss
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_PADDING_INDEX
from torch.utils.checkpoint import checkpoint

from .globals import get_device, set_device, set_cpu_as_device, set_first_gpu, memory, build_cache, get_global

DIR = os.path.dirname(os.path.realpath(__file__))

RE_D = re.compile('\d')


def my_collate(batch):
    # Create and return sample list with proper name and type set
    sample_list = SampleList(batch)
    clean_memory()
    return sample_list


def has_digits(string):
    res = RE_D.search(string)
    return int(res is not None)


def get_all_tags():
    # https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    # https://github.com/nltk/nltk/blob/4e59677df364841c1a23dabfde0317388997aa6d/nltk/sem/relextract.py#L31
    deps = get_universal_deps_indices()
    penn = get_penn_treebank_pos_tag_indices()
    upos = get_pos_tag_indices()
    spacy_glossary = list(glossary.GLOSSARY.keys())

    nltk_ner_tags = ['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION',
                     'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE'] + ['LOC', 'PER', 'ORG'] + ['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION',
                                                                                                   'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE',
                                                                                                   'FACILITY', 'GPE', 'O']
    snlp_list = ["NUMBER", "ORDINAL", "MONEY", "DATE", "TIME", "CAUSE_OF_DEATH", "CITY",
                 "COUNTRY", "CRIMINAL_CHARGE", "EMAIL", "HANDLE", "IDEOLOGY", "NATIONALITY", "RELIGION", "STATE_OR_PROVINCE", "TITLE", "URL"]
    others = ['gsp']
    all_list = deps + penn + upos + spacy_glossary + nltk_ner_tags + snlp_list + others
    tags = list(set(list(map(lambda x: x.lower(), all_list))))
    return dict(zip(tags, range(1, len(tags) + 1)))


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
        if type(sigma) == GaussianNoise:
            sigma = sigma.sigma
        self.sigma = sigma
        self.noise = torch.tensor(0.0, device=get_device())

    def forward(self, x):
        if self.training and self.sigma != 0:
            sigma = self.sigma  # * 1.0/np.sqrt(x.size(-1))
            scale = sigma * x.detach()
            sampled_noise = self.noise.to(x.device).repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def get_regularization_layers(model):
    reg_layers = []
    for c in model.children():
        if isinstance(c, (GaussianNoise, nn.Dropout, nn.Dropout2d, WordMasking)):
            if hasattr(c, "p"):
                reg_layers.append((c, c.p))
            elif hasattr(c, "word_masking_proba"):
                reg_layers.append((c, c.word_masking_proba))
            elif hasattr(c, "sigma"):
                reg_layers.append((c, c.sigma))
            else:
                raise NotImplementedError
        else:
            reg_layers.extend(get_regularization_layers(c))
    return reg_layers


def has_words(text):
    text = re.sub('[ ]+', ' ', text)
    text = re.sub(r"[^A-Za-z ]+", ' ', text)
    tokens = [t for t in text.split() if len(t) >= 3]
    return len(tokens) >= 2


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except:
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


class Transpose(nn.Module):
    def __init__(self, dim1=1, dim2=2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, inp):
        return inp.transpose(self.dim1, self.dim2)


class Average(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inp: torch.Tensor):
        return inp.mean(self.dim)


class ExpandContract(nn.Module):
    def __init__(self, in_dims, out_dims, dropout=0.0, expansion_factor=2,
                 use_layer_norm=False, unit_norm=False,
                 use_layer_norm_input=False, unit_norm_input=False,
                 groups=(2, 4)):
        super().__init__()

        r1 = nn.Conv1d(in_dims, out_dims * expansion_factor, 1, 1, padding=0, groups=groups[0], bias=False)
        init_fc(r1, "leaky_relu")
        r2 = nn.Conv1d(out_dims * expansion_factor, out_dims, 1, 1, padding=0, groups=groups[1], bias=False)
        init_fc(r2, "linear")

        layers = [Transpose(), nn.Dropout(dropout), r1, nn.LeakyReLU(), nn.Dropout(dropout), r2, Transpose()]
        if use_layer_norm_input:
            layers = [nn.LayerNorm(in_dims)] + layers
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dims))
        self.nn = nn.Sequential(*layers)
        self.unit_norm = unit_norm
        self.unit_norm_input = unit_norm_input

    def forward(self, x):
        squeezed = False
        if len(x.size()) < 3:
            assert len(x.size()) == 2
            x = x.unsqueeze(1)
            squeezed = True
        if self.unit_norm_input:
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        x = self.nn(x)
        if squeezed:
            x = x.squeeze()
        if self.unit_norm:
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        return x


def get_torchvision_classification_models(net, large_rf=True, finetune=False):
    from torchvision import models
    sm = 7
    lg = 14
    if "resnet18" in net:
        im_model = models.resnet18(pretrained=True)
        shape = (512, sm, sm) if large_rf else (256, lg, lg)
    elif "resnet34" in net:
        im_model = models.resnet34(pretrained=True)
        shape = (512, sm, sm) if large_rf else (256, lg, lg)
    elif "resnet50" in net:
        im_model = models.resnet50(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnet101" in net:
        im_model = models.resnet101(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnet152" in net:
        im_model = models.resnet152(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)

    elif "resnet18_swsl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
        shape = (512, sm, sm) if large_rf else (256, lg, lg)
    elif "resnet50_swsl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext50_32x4d_swsl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x4d_swsl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x8d_swsl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x16d_swsl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)

    elif "resnet18_ssl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_ssl')
        shape = (512, sm, sm) if large_rf else (256, lg, lg)
    elif "resnet50_ssl" in net:
        im_model = model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext50_32x4d_ssl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x4d_ssl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x8d_ssl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x16d_ssl" in net:
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)

    elif "resnext101_32x8d_wsl" in net:
        im_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x16d_wsl" in net:
        im_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x32d_wsl" in net:
        im_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x48d_wsl" in net:
        im_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)

    elif "mobilenet_v2" in net:
        im_model = models.mobilenet_v2(pretrained=True)
        resnet_layers = im_model.features[:-1] if large_rf else im_model.features[:-5]
        model = nn.Sequential(*resnet_layers)
        shape = (320, sm, sm) if large_rf else (96, lg, lg)
    elif "resnext50_32x4d" in net:
        im_model = models.resnext50_32x4d(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "resnext101_32x8d" in net:
        im_model = models.resnext101_32x8d(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "wide_resnet50_2" in net:
        im_model = models.wide_resnet50_2(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "wide_resnet101_2" in net:
        im_model = models.wide_resnet101_2(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif "mnasnet0_5" in net:
        im_model = models.mnasnet0_5(pretrained=True)
        resnet_layers = im_model.layers[:-3] if large_rf else im_model.layers[:-5]
        model = nn.Sequential(*resnet_layers)
        shape = (160, sm, sm) if large_rf else (48, lg, lg)
    elif "mnasnet1_0" in net:
        im_model = models.mnasnet1_0(pretrained=True)  # models.mnasnet1_0
        resnet_layers = im_model.layers[:-3] if large_rf else im_model.layers[:-5]
        model = nn.Sequential(*resnet_layers)
        shape = (160, sm, sm) if large_rf else (96, lg, lg)
    else:
        raise NotImplementedError(net)

    if "resnet" in net or "resnext" in net:
        resnet_layers = list(im_model.children())[:-3]
        if large_rf:
            resnet_layers = resnet_layers + [im_model.layer4[0], im_model.layer4[1]]
        model = nn.Sequential(*resnet_layers)

    load_stored_params(model, net)

    if not finetune:
        for p in model.parameters():
            p.requires_grad = False

    return model, shape


def get_vgg_face_model(model='resnet'):
    from .resnet50_256 import resnet50_256
    mname = "face_" + model if "face_" not in model else model
    if 'senet' in model:
        raise NotImplementedError
        from .senet50_256 import senet50_256
        model = senet50_256(f"{DIR}/senet50_256.pth")
    elif 'resnet' in model:
        model = resnet50_256(f"{DIR}/resnet50_256.pth")

    for c in list(model.children())[:-1]:
        for p in c.parameters():
            p.requires_grad = False

    for p in list(model.children())[-1].parameters():
        p.requires_grad = True

    model = nn.Sequential(model, LambdaLayer(lambd=lambda x: x[1].squeeze(2).transpose(1, 2)), )
    if mname + ".pth" in os.listdir("."):
        print("Loading saved model: ", mname + ".pth")
        model.load_state_dict(torch.load(mname + ".pth"))

    load_stored_params(model, mname)
    return model


def load_stored_params(model, key):
    if key + ".pth" in os.listdir("."):
        print("Loading saved model: ", key + ".pth")
        model.load_state_dict(torch.load(key + ".pth"))

    if key in os.listdir("."):
        print("Loading saved model: ", key)
        model.load_state_dict(torch.load(key))

    global_dir = get_global("models_dir")
    if key + ".pth" in os.listdir(global_dir):
        print("Loading saved model: ", key + ".pth")
        model.load_state_dict(torch.load(os.path.join(global_dir, key + ".pth")))

    if key in os.listdir(global_dir):
        print("Loading saved model: ", key)
        model.load_state_dict(torch.load(os.path.join(global_dir, key)))


def save_params(model, key):
    key = key if ".pth" in key else key + ".pth"
    global_dir = get_global("models_dir")
    torch.save(model.state_dict(), os.path.join(global_dir, key))


def loss_calculator(logits, labels, task, loss_fn):
    logits = logits.to(get_device())
    loss = torch.tensor(0.0, device=get_device())
    if labels is not None:
        labels = labels.to(get_device())
        if task == "classification" or task == "focal":
            assert len(labels.size()) == 1
            loss = loss_fn(logits, labels.long())
            logits = torch.softmax(logits, dim=1)
        elif task == "regression":
            assert len(labels.size()) == 2
            loss = loss_fn(logits, labels.float())

    if task == "classification" or task == "focal":
        logits = torch.softmax(logits, dim=1)

    return logits, loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True, sentinel_class=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.sentinel_class = sentinel_class

    def forward(self, inputs, targets):
        if self.sentinel_class is not None:
            mult = targets != self.sentinel_class
            inputs = inputs[mult]
            targets = targets[mult]
        BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class CELoss(nn.Module):
    def __init__(self, reduce=True, sentinel_class=None):
        super().__init__()
        self.reduce = reduce
        self.sentinel_class = sentinel_class

    def forward(self, inputs, targets):
        if self.sentinel_class is not None:
            mult = targets != self.sentinel_class
            inputs = inputs[mult]
            targets = targets[mult]
        BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        if self.reduce:
            return torch.mean(BCE_loss)
        else:
            return BCE_loss


def get_loss_by_task(task, n_classes):
    if callable(task):
        raise NotImplementedError
        loss = task
    elif task == "classification":
        loss = CELoss(sentinel_class=n_classes)
    elif task == "focal":
        loss = FocalLoss(sentinel_class=n_classes)
    elif task == "regression":
        loss = nn.MSELoss()
    elif task == "k-classification":
        loss = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

    return loss


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding2D(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, channels_first=False):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.d_model = d_model
        self.channels_first = channels_first

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the 2D image features fed to the positional encoder model (required).
        Shape:
            x: [batch size, H, W, C]
            output: [HxW, batch size, C]
        Examples:
            >>> output = pos_encoder(x)
        """
        b = x.size(0)
        if self.channels_first:
            x = x.transpose(1, 2).transpose(2, 3)
        assert x.size(-1) == self.d_model
        x = x.transpose(0, 1).transpose(1, 2)  # H, W, B, C
        pe = self.pe[:x.size(0), :]  # H, C
        pe_abs = self.pe[:x.size(0) * x.size(1), :]
        pe2 = self.pe[:x.size(1), :]  # W, C
        pe1 = pe.unsqueeze(1)
        pe2 = pe2.unsqueeze(0)
        x = x + 0.3 * pe1
        x = x + 0.3 * pe2
        x = x.flatten(0, 1) + pe_abs / 3
        return self.dropout(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, gaussian_noise=0.0, attention_drop_proba=0.0):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.gaussian_noise = GaussianNoise(gaussian_noise)
        self.attention_drop_proba = attention_drop_proba

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            if self.attention_drop_proba > 0 and self.training:
                mask = torch.rand((src.size(0), src.size(0)))
                drops = mask <= self.attention_drop_proba
                keeps = mask > self.attention_drop_proba
                mask[drops] = -1.0e4
                mask[keeps] = 0.0
                mask = mask.to(get_device())
            output = mod(self.gaussian_noise(output), src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, gaussian_noise=0.0, attention_drop_proba=0.0):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.gaussian_noise = GaussianNoise(gaussian_noise)
        self.attention_drop_proba = attention_drop_proba

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            if self.attention_drop_proba > 0 and self.training:
                memory_mask = torch.rand((tgt.size(0), memory.size(0)))
                drops = memory_mask <= self.attention_drop_proba
                keeps = memory_mask > self.attention_drop_proba
                memory_mask[drops] = -1.0e4
                memory_mask[keeps] = 0.0
                memory_mask = memory_mask.to(get_device())

                tgt_mask = torch.rand((tgt.size(0), tgt.size(0)))
                drops = tgt_mask <= self.attention_drop_proba
                keeps = tgt_mask > self.attention_drop_proba
                tgt_mask[drops] = -1.0e4
                tgt_mask[keeps] = 0.0
                tgt_mask = tgt_mask.to(get_device())

            output = mod(self.gaussian_noise(output), self.gaussian_noise(memory), tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer(nn.Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, gaussian_noise: float = 0.0, attention_drop_proba=0.0,
                 activation: str = "relu") -> None:
        super(Transformer, self).__init__()
        assert num_encoder_layers > 0 or num_decoder_layers > 0
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.gaussian_noise = GaussianNoise(gaussian_noise)

        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, gaussian_noise, attention_drop_proba)

        if num_decoder_layers > 0:
            decoder_norm = LayerNorm(d_model)
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, gaussian_noise, attention_drop_proba)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        if src.size(2) != self.d_model:
            raise RuntimeError("the feature number of src must be equal to d_model")

        memory = src
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        output = memory
        if self.num_decoder_layers > 0:
            if src.size(1) != tgt.size(1):
                raise RuntimeError("the batch number of src and tgt must be equal")
            if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
                raise RuntimeError("the feature number of src and tgt must be equal to d_model")
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        return output, memory

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class MultiLayerTransformerDecoderHead(nn.Module):
    def __init__(self, n_dims, n_tokens, n_out, dropout, gaussian_noise,
                 attention_drop_proba,
                 loss=None, n_queries=16, n_layers=3, n_decoders=2, **kwargs):
        super().__init__()
        self.task = loss
        if loss not in ["classification", "focal", "regression", "k-classification"]:
            raise NotImplementedError(loss)

        decoders = nn.ModuleList()
        classifiers = nn.ModuleList()
        decoder_queries = nn.ParameterList()
        tgt_norms = nn.ModuleList()

        decoder_layer = TransformerDecoderLayer(n_dims, 16, n_dims * 4, dropout, "relu")
        for i in range(n_decoders):
            decoder_norm = LayerNorm(n_dims)
            decoder = TransformerDecoder(decoder_layer, n_layers, decoder_norm, gaussian_noise, attention_drop_proba)
            decoders.append(decoder)
            classifier = DecoderEnsemblingHead(n_dims, n_queries, n_out, dropout, loss, **kwargs)
            classifiers.append(classifier)
            decoder_query = nn.Parameter(torch.randn(n_queries, n_dims) * (1 / n_dims), requires_grad=True)
            tgt_norm = nn.LayerNorm(n_dims)
            decoder_queries.append(decoder_query)
            tgt_norms.append(tgt_norm)

        self.decoders = decoders
        self.tgt_norms = tgt_norms
        self.decoder_queries = decoder_queries
        self.classifiers = classifiers
        self.n_tokens, self.n_dims, self.n_out, self.n_layers, self.n_decoders = n_tokens, n_dims, n_out, n_layers, n_decoders
        self.gaussian_noise = GaussianNoise(gaussian_noise)
        self.global_layer_norm = nn.LayerNorm(self.n_dims)
        self.pos_encoder = PositionalEncoding(self.n_dims)

        self._reset_parameters()

    def forward(self, x, labels=None):
        x = x.transpose(0, 1) * math.sqrt(self.n_dims)
        x = self.pos_encoder(x)
        x = self.global_layer_norm(x)
        batch_size = x.size(1)
        # transformer_tgt = self.pos_encoder(transformer_tgt)
        # transformer_tgt = self.tgt_norm(transformer_tgt) # R
        losses, logits = [], []
        for i in range(self.n_decoders):
            decoder = self.decoders[i]
            classifier = self.classifiers[i]
            decoder_query = self.decoder_queries[i]
            tgt_norm = self.tgt_norms[i]
            # TODO: TGT norm test here with pos encoding

            transformer_tgt = decoder_query.unsqueeze(0).expand(batch_size, *decoder_query.size())
            transformer_tgt = transformer_tgt.transpose(0, 1) * math.sqrt(self.n_dims)
            transformer_tgt = self.pos_encoder(transformer_tgt)
            transformer_tgt = tgt_norm(transformer_tgt)
            transformer_tgt = decoder(transformer_tgt, x).transpose(0, 1)

            logit, loss = classifier(transformer_tgt, labels)
            losses.append(loss)
            logits.append(logit)
        loss = torch.stack(losses).mean()
        logits = torch.stack(logits).mean(0)
        return logits, loss

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class GRUHead:
    pass


class CNNHead(nn.Module):
    def __init__(self, n_dims, n_tokens, n_out, dropout,
                 loss, width="wide", **kwargs):
        super().__init__()
        uda = kwargs["uda"] if "uda" in kwargs else False
        if loss not in ["classification", "focal", "regression", "k-classification"]:
            raise NotImplementedError(loss)
        self.task = loss
        self.loss = get_loss_by_task(loss, n_out if uda else None)
        c1 = nn.Conv1d(n_dims, n_out, 3 if width == "narrow" else n_tokens, 1, padding=0, groups=1, bias=True)
        init_fc(c1, "linear")
        avp = nn.AdaptiveAvgPool1d(1)
        dp = nn.Dropout(dropout)
        self.classifier = nn.Sequential(dp, Transpose(), c1, avp)
        self.n_tokens, self.n_dims, self.n_out = n_tokens, n_dims, n_out

    def forward(self, x, labels=None):
        """

        :param x: Final Features in shape: (Batch, Seq, Embedding_dims)
        :param labels: task specific labels with shape: (Batch,) for classification and (Batch,*) for regression and k-classification
        :return: loss, logits
        """
        logits = self.classifier(x).squeeze()
        logits = logits.to(get_device())
        return loss_calculator(logits, labels if self.training else None, self.task, self.loss)


class DecoderEnsemblingHead(nn.Module):
    """
    We asume that each token coming into this head is coming from a Transformer Decoder.
    """

    def __init__(self, n_dims, n_tokens, n_out, dropout,
                 loss, **kwargs):
        super().__init__()
        if loss not in ["classification", "focal", "regression", "k-classification"]:
            raise NotImplementedError(loss)
        self.task = loss
        uda = kwargs["uda"] if "uda" in kwargs else False
        self.loss = get_loss_by_task(loss, n_out if uda else None)
        n_classifier_layers = kwargs["n_classifier_layers"] if "n_classifier_layers" in kwargs else 1
        n_classifiers = kwargs["n_classifiers"] if "n_classifiers" in kwargs else 2
        assert n_classifiers <= n_tokens
        assert n_classifier_layers in [1, 2]
        classifiers = nn.ModuleList()
        for i in range(n_classifiers + 1):
            classifier = LinearHead(n_dims, n_tokens, n_out, dropout, loss, **kwargs)
            if i == n_classifiers:
                classifier = CNNHead(n_dims, n_tokens, n_out, dropout, loss, **kwargs)
            classifiers.append(classifier)

        self.classifiers = classifiers
        self.n_tokens, self.n_dims, self.n_out = n_tokens, n_dims, n_out

    def forward(self, x, labels=None):
        assert len(x.size()) == 3 and x.size()[1:] == (self.n_tokens, self.n_dims)
        losses, logits, weights = [], [], []
        for i, classifier in enumerate(self.classifiers):
            if i == len(self.classifiers) - 1:
                logit, loss = classifier(x, labels if self.training else None)
                weights.append(1.0)
            else:
                tokens = x[:, i].squeeze()
                logit, loss = classifier(tokens, labels if self.training else None)
                weights.append(0.1)  # TODO check other weights usefulness here
            losses.append(loss)
            logits.append(logit)
        ws = sum(weights)
        loss = torch.stack([w * l / ws for w, l in zip(weights, losses)]).sum()
        logits = torch.stack([w * l / ws for w, l in zip(weights, logits)]).sum(0)
        return logits, loss


class LinearHead(CNNHead):
    def __init__(self, n_dims, n_tokens, n_out, dropout,
                 loss, **kwargs):
        """
        Expected input in format (Batch, Seq, Embedding_dims)
        :param n_dims: Embedding_dims
        :param n_tokens: Sequence Length
        :param n_out:
        :param dropout:
        :param task:
        :param loss:
        """
        super().__init__(n_dims, n_tokens, n_out, dropout, loss, **kwargs)
        n_classifier_layers = kwargs["n_classifier_layers"] if "n_classifier_layers" in kwargs else 1
        lin0 = nn.Linear(n_dims, n_dims)
        init_fc(lin0, "leaky_relu")
        lin = nn.Linear(n_dims, n_out)
        init_fc(lin, "linear")
        dp = nn.Dropout(dropout)
        self.classifier = nn.Sequential(dp, lin)
        if n_classifier_layers == 2:
            self.classifier = nn.Sequential(dp, lin0, nn.LeakyReLU(), lin)


class LambdaLayer(nn.Module):
    def __init__(self, lambd, gaussian_noise=0.0, dropout=0.0):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        self.gaussian_noise = GaussianNoise(gaussian_noise)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        x = self.lambd(x, *args, **kwargs)
        x = x.to(get_device())
        x = self.dropout(x)
        x = self.gaussian_noise(x)
        return x


from .detectron_v1_object_detector import get_image_info_fn, persistent_caching_fn


def print_code(func):
    import inspect
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import TerminalFormatter

    code = "".join(inspect.getsourcelines(func)[0])
    print(highlight(code, PythonLexer(), TerminalFormatter()))


def dict2sampleList(samples: Dict, device: torch.device):
    if type(samples) == dict:
        sl = SampleList()
        for k, v in samples.items():
            assert type(k) == str or type(k) == tuple
            assert type(v) == list or type(v) == torch.Tensor or type(v) == str
            sl[k] = v
        return sl
    elif type(samples) == SampleList:
        return samples
    elif type(samples) == list and type(samples[0]) == Sample:
        return SampleList(samples)
    else:
        raise ValueError


def clean_memory():
    _ = gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _ = gc.collect()


class WordMasking(nn.Module):
    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.word_masking_proba = kwargs.pop("word_masking_proba", 0.0)
        self.whole_word_masking = kwargs.pop("whole_word_masking", False)

    def forward(self, texts):
        if self.training and self.word_masking_proba > 0:
            tokenizer = self.tokenizer
            proba = self.word_masking_proba
            if self.whole_word_masking:
                texts = [random_whole_word_mask(t, tokenizer, proba) for t in texts]
            else:
                texts = [random_word_mask(t, tokenizer, proba) for t in texts]
        return texts


def random_whole_word_mask(text: str, tokenizer, probability: float) -> str:
    text = str(text)
    if probability == 0 or len(text) == 0 or len(text.split()) <= 2:
        return text
    tokens = text.split()
    new_tokens = []
    for idx, token in enumerate(tokens):
        prob = random.random()
        if prob < probability:
            prob /= probability
            if prob < 0.8 or len(token) <= 3:
                tks = [tokenizer.mask_token] * len(tokenizer.tokenize(token))
            else:
                tks = [token]
        else:
            tks = [token]
        new_tokens.extend(tks)
    return " ".join(new_tokens)


def random_word_mask(text: str, tokenizer, probability: float) -> str:
    text = str(text)
    if probability == 0 or len(text.split()) <= 1:
        return text
    tokens = tokenizer.tokenize(text)
    for idx, token in enumerate(tokens):
        prob = random.random()

        if prob < probability:
            prob /= probability

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[idx] = tokenizer.mask_token
            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[idx] = tokenizer.convert_ids_to_tokens(
                    torch.randint(len(tokenizer), (1,), dtype=torch.long)
                )[0]

            # rest 10% keep the original token as it is
    return tokenizer.convert_tokens_to_string(tokens)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            from transformers.modeling_bert import ACT2FN
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act, n_tokens_in):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.vocab_size = vocab_size

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias
        self.n_tokens_in = n_tokens_in
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def forward(self, hidden_states, input_ids, attention_mask):
        hidden_states = hidden_states[:, :self.n_tokens_in]
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        hidden_states = hidden_states.view(-1, self.vocab_size)
        input_ids = input_ids.view(-1)
        masked_lm_loss = 0.0
        if self.training:
            masked_lm_loss = self.loss_fct(hidden_states, input_ids)
            # attention_mask = attention_mask.view(-1)
            # masked_lm_loss = attention_mask * masked_lm_loss
            masked_lm_loss = masked_lm_loss.mean()

        predictions = hidden_states.max(dim=1).indices
        accuracy = accuracy_score(input_ids.cpu(), predictions.cpu())
        return masked_lm_loss, accuracy, input_ids, predictions


class MLMPretraining(nn.Module):
    def __init__(self, model, tokenizer, hidden_size, hidden_act, n_tokens_in, use_as_super=False):
        super().__init__()
        self.model = model

        if not use_as_super:
            self.mlm = BertLMPredictionHead(hidden_size, tokenizer.vocab_size, hidden_act, n_tokens_in)
            self.tokenizer = tokenizer
            self.n_tokens_in = n_tokens_in
        self.accuracy_hist = []
        self.loss_hist = []

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.n_tokens_in
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def forward(self, samples: SampleList):
        _, pooled, seq, _ = self.model(samples)
        text = samples["text"]
        input_ids, attention_mask = self.tokenise(text)
        loss, accuracy, input_ids, predictions = self.mlm(seq, input_ids, attention_mask)
        self.accuracy_hist.append(accuracy)
        self.loss_hist.append(float(loss.cpu().detach()))
        return [accuracy, input_ids, predictions, loss]

    def plot_loss_acc_hist(self):
        import matplotlib.pyplot as plt
        t = list(range(1, len(self.loss_hist) + 1))
        fig, ax1 = plt.subplots(figsize=(8, 8))

        color = 'tab:red'
        ax1.set_xlabel('Training Batches')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(t, self.loss_hist, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, self.accuracy_hist, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def test_accuracy(self, batch_size, dataset, collate_fn=my_collate):
        from tqdm.auto import tqdm as tqdm
        test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                                 shuffle=False, num_workers=get_global("dataloader_workers"), pin_memory=True)

        use_autocast = False
        try:
            from torch.cuda.amp import autocast
            use_autocast = "cuda" in str(get_device())
        except:
            pass
        use_autocast = use_autocast and get_global("use_autocast")
        labels_list = []
        predictions_list = []
        with torch.no_grad():
            clean_memory()
            with tqdm(test_loader, "Test MLM Accuracy") as test_loader:
                for batch in test_loader:
                    if use_autocast:
                        with autocast():
                            accuracy, labels, predictions, loss = self(batch)
                    else:
                        accuracy, labels, predictions, loss = self(batch)

                    labels_list.extend(labels.tolist() if type(labels) == torch.Tensor else labels)
                    predictions = predictions.cpu().detach()
                    predictions_list.extend(predictions.tolist())
            accuracy = accuracy_score(labels_list, predictions_list)
            print("MLM Accuracy = %.4f" % accuracy)
            clean_memory()
            return accuracy


class SimCLR(MLMPretraining):
    def __init__(self, model, in_dims, hidden_size, dropout, augment_1: Callable, augment_2: Callable, low_memory=False):
        super(SimCLR, self).__init__(model, None, hidden_size, "leaky_relu", 0, True)
        self.aug_1 = augment_1
        self.aug_2 = augment_2
        self.model = model
        self.aug_time = []
        self.model_time = []
        self.low_memory = low_memory

        lin0 = nn.Linear(in_dims, hidden_size)
        init_fc(lin0, "leaky_relu")
        lin = nn.Linear(hidden_size, hidden_size)
        init_fc(lin, "linear")
        self.final_layer = nn.Sequential(nn.Dropout(dropout), lin0, nn.LeakyReLU(), lin)
        self.loss = nn.CrossEntropyLoss()
        # TODO: Add temperature parameter

    def forward(self, x):
        ats = time.time()
        x1 = self.aug_1(x)
        x2 = self.aug_2(x)
        mts = time.time()
        self.aug_time.append(mts - ats)
        if self.low_memory and hasattr(x1, "__len__"):
            x1 = checkpoint(self.model, x1)
            clean_memory()
            x2 = checkpoint(self.model, x2)
            clean_memory()
        else:
            x1 = self.model(x1)
            x2 = self.model(x2)
        self.model_time.append(time.time() - mts)

        if isinstance(x1, (list, tuple)):
            x1 = [(len(x.size()), x) for x in x1 if isinstance(x, torch.Tensor)]
            x2 = [(len(x.size()), x) for x in x2 if isinstance(x, torch.Tensor)]
            x1 = list(sorted(x1, key=operator.itemgetter(0), reverse=True))[0][1]
            x2 = list(sorted(x2, key=operator.itemgetter(0), reverse=True))[0][1]
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        x1 = self.final_layer(x1)
        x2 = self.final_layer(x2)

        if len(x1.size()) == 3:
            x1 = x1 / x1.norm(dim=2, keepdim=True).clamp(min=1e-5)
            x2 = x2 / x2.norm(dim=2, keepdim=True).clamp(min=1e-5)
            if x1.size(1) > 16:
                x1 = torch.cat((x1[:, :16].flatten(1, 2).squeeze(), x1[:, 16:].mean(1)), 1)
                x2 = torch.cat((x2[:, :16].flatten(1, 2).squeeze(), x2[:, 16:].mean(1)), 1)
            else:
                x1 = x1.flatten(1, 2).squeeze()
                x2 = x2.flatten(1, 2).squeeze()

        x1 = x1 / x1.norm(dim=1, keepdim=True).clamp(min=1e-5)
        x2 = x2 / x2.norm(dim=1, keepdim=True).clamp(min=1e-5)
        x2 = x2.transpose(0, 1)
        x = x1.mm(x2)  # batch x batch
        labels = torch.arange(0, len(x), device=x.device, dtype=torch.long)
        loss = self.loss(x, labels)
        x = torch.softmax(x, 1)
        predictions = x.max(dim=1).indices
        accuracy = accuracy_score(labels.cpu(), predictions.cpu())
        self.accuracy_hist.append(accuracy)
        self.loss_hist.append(float(loss.cpu().detach()))
        return [accuracy, labels, predictions, loss]

    def plot_timing(self):
        import matplotlib.pyplot as plt
        t = list(range(1, len(self.aug_time) + 1))
        fig, ax1 = plt.subplots(figsize=(8, 8))

        color = 'tab:red'
        ax1.set_xlabel('Augment time per batch Batches')
        ax1.set_ylabel('Augment Time', color=color)
        ax1.plot(t, self.aug_time, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Model Runtime per batch', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, self.model_time, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


def merge_sample_lists(*samples):
    nsl = SampleList()
    fields = samples[0].keys()
    for field in fields:
        if isinstance(samples[0][field], torch.Tensor):
            nsl[field] = torch.cat([s[field] for s in samples], 0)
        elif isinstance(samples[0][field], SampleList):
            nsl[field] = merge_sample_lists(*[s[field] for s in samples])
        else:
            nsl[field] = [f for s in samples for f in s[field]]
    return nsl


def run_simclr(smclr, pre_dataset, post_dataset, lr_strategy_pre, lr_strategy_post,
               pre_lr, post_lr, pre_batch_size, post_batch_size,
               pre_epochs, full_epochs, collate_fn, scheduler_init_fn=None, test_acc=False, effective_batch_size=256, sampling_policy=None,):
    from ..training import group_wise_finetune, group_wise_lr, train, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
    acc_head = np.nan
    if pre_epochs > 0:
        epochs = pre_epochs
        optimizer_class = torch.optim.AdamW
        optimizer_params = dict(lr=pre_lr,
                                betas=(0.9, 0.98),
                                eps=1e-08,
                                weight_decay=1e-2)

        _ = group_wise_finetune(smclr, lr_strategy_pre)
        params_conf, _ = group_wise_lr(smclr, lr_strategy_pre)
        optimizer = optimizer_class(params_conf, **optimizer_params)
        train_losses, learning_rates, _ = train(smclr,
                                                optimizer,
                                                scheduler_init_fn,
                                                pre_batch_size,
                                                epochs,
                                                pre_dataset,
                                                model_call_back=None,
                                                accumulation_steps=effective_batch_size // pre_batch_size + 1,
                                                plot=True,
                                                collate_fn=collate_fn,
                                                sampling_policy=sampling_policy,
                                                class_weights=None)

        if hasattr(smclr, "plot_loss_acc_hist"):
            smclr.plot_loss_acc_hist()
        if test_acc and hasattr(smclr, "test_accuracy"):
            acc_head = smclr.test_accuracy(pre_batch_size, pre_dataset, collate_fn=collate_fn)

    ##

    epochs = full_epochs
    optimizer_class = torch.optim.AdamW
    optimizer_params = dict(lr=post_lr,
                            betas=(0.9, 0.98),
                            eps=1e-08,
                            weight_decay=1e-3)

    _ = group_wise_finetune(smclr, lr_strategy_post)
    params_conf, _ = group_wise_lr(smclr, lr_strategy_post)
    optimizer = optimizer_class(params_conf, **optimizer_params)
    train_losses, learning_rates, _ = train(smclr,
                                            optimizer,
                                            scheduler_init_fn,
                                            post_batch_size,
                                            epochs,
                                            post_dataset,
                                            model_call_back=None,
                                            accumulation_steps=effective_batch_size // post_batch_size + 1,
                                            plot=True,
                                            collate_fn=collate_fn,
                                            sampling_policy=sampling_policy,
                                            class_weights=None)

    if hasattr(smclr, "plot_loss_acc_hist"):
        smclr.plot_loss_acc_hist()
    acc = np.nan
    if test_acc and hasattr(smclr, "test_accuracy"):
        acc = smclr.test_accuracy(post_batch_size, post_dataset, collate_fn=collate_fn)
        print("Head Acc = ", acc_head, "Full Acc = ", acc)
    return (acc_head, acc)
