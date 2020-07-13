import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmf.common import SampleList, Sample
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import contractions
import pandas as pd
import jsonlines
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_PADDING_INDEX
from spacy import glossary
from .globals import get_device, set_device, set_cpu_as_device, set_first_gpu, memory, build_cache
import os
import torch
import gc
import os
import random
DIR = os.path.dirname(os.path.realpath(__file__))


RE_D = re.compile('\d')


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
        self.sigma = sigma
        self.noise = torch.tensor(0.0, device=get_device())

    def forward(self, x):
        if self.training and self.sigma != 0:
            sigma = self.sigma  # * 1.0/np.sqrt(x.size(-1))
            scale = sigma * x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


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


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.transpose(1, 2)


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
    sm = 10
    lg = 20
    if net == "resnet18":
        im_model = models.resnet18(pretrained=True)
        shape = (512, sm, sm) if large_rf else (256, lg, lg)
    elif net == "resnet34":
        im_model = models.resnet34(pretrained=True)
        shape = (512, sm, sm) if large_rf else (256, lg, lg)
    elif net == "resnet50":
        im_model = models.resnet50(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnet101":
        im_model = models.resnet101(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnet152":
        im_model = models.resnet152(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)

    elif net == "resnet18_swsl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
        shape = (512, sm, sm) if large_rf else (256, lg, lg)
    elif net == "resnet50_swsl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext50_32x4d_swsl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x4d_swsl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x8d_swsl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x16d_swsl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)

    elif net == "resnet18_ssl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_ssl')
        shape = (512, sm, sm) if large_rf else (256, lg, lg)
    elif net == "resnet50_ssl":
        im_model = model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext50_32x4d_ssl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x4d_ssl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x8d_ssl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x16d_ssl":
        im_model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_ssl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)

    elif net == "resnext101_32x8d_wsl":
        im_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x16d_wsl":
        im_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x32d_wsl":
        im_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x48d_wsl":
        im_model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)

    elif net == "mobilenet_v2":
        im_model = models.mobilenet_v2(pretrained=True)
        resnet_layers = im_model.features[:-1] if large_rf else im_model.features[:-5]
        model = nn.Sequential(*resnet_layers)
        shape = (320, sm, sm) if large_rf else (96, lg, lg)
    elif net == "resnext50_32x4d":
        im_model = models.resnext50_32x4d(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "resnext101_32x8d":
        im_model = models.resnext101_32x8d(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "wide_resnet50_2":
        im_model = models.wide_resnet50_2(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "wide_resnet101_2":
        im_model = models.wide_resnet101_2(pretrained=True)
        shape = (2048, sm, sm) if large_rf else (1024, lg, lg)
    elif net == "mnasnet0_5":
        im_model = models.mnasnet0_5(pretrained=True)
        resnet_layers = im_model.layers[:-3] if large_rf else im_model.layers[:-5]
        model = nn.Sequential(*resnet_layers)
        shape = (160, sm, sm) if large_rf else (48, lg, lg)
    elif net == "mnasnet1_0":
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

    if net+".pth" in os.listdir("."):
        print("Loading saved model: ", net+".pth")
        model.load_state_dict(torch.load(net+".pth"))

    if not finetune:
        for p in model.parameters():
            p.requires_grad = False

    return model, shape


def get_vgg_face_model(model='resnet'):
    from .senet50_256 import senet50_256
    from .resnet50_256 import resnet50_256
    mname = "face_" + model
    if model == 'senet':
        raise NotImplementedError
        model = senet50_256(f"{DIR}/senet50_256.pth")
    elif model == 'resnet':
        model = resnet50_256(f"{DIR}/resnet50_256.pth")

    for c in list(model.children())[:-1]:
        for p in c.parameters():
            p.requires_grad = False

    model = nn.Sequential(model, LambdaLayer(lambd=lambda x: x[1].squeeze(2).transpose(1, 2)),)
    if mname+".pth" in os.listdir("."):
        print("Loading saved model: ", mname+".pth")
        model.load_state_dict(torch.load(mname+".pth"))
    return model


def loss_calculator(logits, labels, task, loss_fn):
    logits = logits.to(get_device())
    loss = torch.tensor(0.0, device=get_device())
    if labels is not None:
        labels = labels.to(get_device())
        if task == "classification":
            assert len(labels.size()) == 1
            loss = loss_fn(logits, labels.long())
            # preds = logits.max(dim=1).indices
            logits = torch.softmax(logits, dim=1)
        elif task == "regression":
            assert len(labels.size()) == 2
            loss = loss_fn(logits, labels.float())

        elif task == "k-classification":
            assert len(labels.size()) == 2
            labels = labels.astype(float)
            loss = loss_fn(logits, labels)
            logits = torch.sigmoid(logits)

    if task == "classification":
        logits = torch.softmax(logits, dim=1)
    elif task == "k-classification":
        logits = torch.sigmoid(logits)

    return logits, loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def get_loss_by_task(task):
    if task == "classification":
        # loss = nn.CrossEntropyLoss()
        loss = FocalLoss()
    elif task == "regression":
        loss = nn.MSELoss()
    elif task == "k-classification":
        loss = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

    return loss


class CNNHead(nn.Module):
    def __init__(self, n_dims, n_tokens, n_out, dropout,
                 task, loss=None, width="wide", ):
        super().__init__()
        if task not in ["classification", "regression", "k-classification"]:
            raise NotImplementedError(task)
        # TODO: Implement n_of_k class classification or set prediction/bipartite loss
        self.task = task
        self.loss = get_loss_by_task(task)

        if loss is not None:
            self.loss = loss

        c1 = nn.Conv1d(n_dims, n_out, 3 if width == "narrow" else n_tokens, 1, padding=0, groups=1, bias=False)
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
        assert ((len(x.size()) == 3 and x.size()[1:] == (self.n_tokens, self.n_dims))
                or (len(x.size()) == 4 and x.size()[1] == self.n_dims))
        logits = self.classifier(x).squeeze()
        logits = logits.to(get_device())
        return loss_calculator(logits, labels, self.task, self.loss)


class CNN2DHead(CNNHead):
    def __init__(self, n_dims, n_out, dropout,
                 task, loss=None, ):
        super().__init__(n_dims, 1, n_out, dropout,
                         task, loss)

        conv = nn.Conv2d(n_dims, n_out, 3)
        init_fc(conv, "linear")
        dp = nn.Dropout(dropout)
        self.classifier = nn.Sequential(dp, conv, nn.AdaptiveAvgPool2d(1))


class AveragedLinearHead(CNNHead):
    def __init__(self, n_dims, n_tokens, n_out, dropout,
                 task, loss=None, ):
        """
        Expected input in format (Batch, Seq, Embedding_dims)
        :param n_dims: Embedding_dims
        :param n_tokens: Sequence Length
        :param n_out:
        :param dropout:
        :param task:
        :param loss:
        """
        super().__init__(n_dims, n_tokens, n_out, dropout,
                         task, loss)
        lin0 = nn.Linear(n_dims, n_dims)
        init_fc(lin0, "leaky_relu")
        lin = nn.Linear(n_dims, n_out)
        init_fc(lin, "linear")
        dp = nn.Dropout(dropout)
        ll = nn.LayerNorm(n_dims)
        self.classifier = nn.Sequential(dp, Average(1), lin0, nn.LeakyReLU(), ll, lin)


class LinearHead(CNNHead):
    def __init__(self, n_dims, n_tokens, n_out, dropout,
                 task, loss=None, ):
        """
        Expected input in format (Batch, Seq, Embedding_dims)
        :param n_dims: Embedding_dims
        :param n_tokens: Sequence Length
        :param n_out:
        :param dropout:
        :param task:
        :param loss:
        """
        super().__init__(n_dims, n_tokens, n_out, dropout,
                         task, loss)
        lin0 = nn.Linear(n_dims, n_dims)
        init_fc(lin0, "leaky_relu")
        lin = nn.Linear(n_dims, n_out)
        init_fc(lin, "linear")
        dp = nn.Dropout(dropout)
        ll = nn.LayerNorm(n_dims)
        self.classifier = nn.Sequential(ll, dp, lin0, nn.LeakyReLU(), lin)


class PositionExtract(nn.Module):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos

    def forward(self, inp):
        return inp[:, self.pos].squeeze()


class OneTokenPositionLinearHead(nn.Module):
    def __init__(self, n_dims, n_tokens, n_out, dropout,
                 task, loss=None, extract_pos=0):
        super().__init__(n_dims, n_tokens, n_out, dropout,
                         task, loss)
        lin0 = nn.Linear(n_dims, n_dims)
        init_fc(lin0, "leaky_relu")
        lin = nn.Linear(n_dims, n_out)
        init_fc(lin, "linear")
        dp = nn.Dropout(dropout)
        ll = nn.LayerNorm(n_dims)
        self.classifier = nn.Sequential(dp, PositionExtract(extract_pos), lin0, nn.LeakyReLU(), ll, lin)


class MultiTaskForward(nn.Module):
    def __init__(self, task_heads: List, task_weights=None):
        super().__init__()
        self.heads = nn.ModuleList(task_heads)
        assert task_weights is None or len(task_weights) == len(task_heads)
        if task_weights is None:
            task_weights = torch.ones(len(task_heads), dtype=float, device=get_device())  # use device= param for directly creating on target device
        self.task_weights = task_weights.to(get_device())

    def forward(self, x, labels=None):
        assert labels is None or len(labels) == len(self.heads) or len(self.heads) == 1
        logits_list = []
        loss_total = torch.tensor(0.0, device=get_device())
        if len(self.heads) == 1 and type(labels) not in [list, tuple]:
            labels = [labels]

        for i, m in enumerate(self.heads):
            logits, loss = m(x, labels[i])
            logits_list.append(logits)
            loss_total += loss * self.task_weights[i]

        return logits_list if len(logits_list) > 1 else logits_list[0], loss_total


class LambdaLayer(nn.Module):
    def __init__(self, lambd, gaussian_noise=0.0, dropout=0.0):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        self.gaussian_noise = GaussianNoise(gaussian_noise)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lambd(x)
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


def random_word_mask(text: str, tokenizer, probability: float) -> str:
    if probability == 0:
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
