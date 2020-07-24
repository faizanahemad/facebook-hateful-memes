from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from .BaseFeaturizer import BaseFeaturizer
from ...utils import init_fc, GaussianNoise, ExpandContract, LambdaLayer, get_device, PositionalEncoding, PositionalEncoding2D, Transformer
import math
import sys


class TransformerFeaturizer(nn.Module):
    def __init__(self, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_encoders, n_decoders,
                 n_decoder_ensembles=1,
                 gaussian_noise=0.0, dropout=0.0, attention_drop_proba=0.0):
        super(TransformerFeaturizer, self).__init__()
        assert n_tokens_out % n_decoder_ensembles == 0
        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        self.n_tokens_out = n_tokens_out
        self.n_channels_out = n_channels_out
        self.n_internal_dims = n_internal_dims
        self.n_decoders = n_decoders
        self.transformer_needed = n_encoders > 0 or n_decoders > 0
        if not self.transformer_needed:
            return
        if n_decoders > 0:
            decoder_query = nn.Parameter(torch.randn((n_tokens_out, n_internal_dims)) * (1 / n_internal_dims),
                                         requires_grad=True)
            self.register_parameter("decoder_query", decoder_query)

        self.input_nn = None
        if n_channels_in != n_internal_dims:
            self.input_nn = nn.Linear(n_channels_in, n_internal_dims, bias=False)
            init_fc(self.input_nn, "linear")

        self.output_nn = None
        if n_internal_dims != n_channels_out:
            self.output_nn = nn.Linear(n_internal_dims, n_channels_out, bias=False)
            init_fc(self.output_nn, "linear")

        self.transformer = Transformer(n_internal_dims, 8, n_encoders, n_decoders, n_internal_dims*4, dropout, gaussian_noise, attention_drop_proba)
        self.pos_encoder = PositionalEncoding(n_internal_dims, dropout)
        self.global_layer_norm = nn.LayerNorm(n_internal_dims)

    def forward(self, x):
        if not self.transformer_needed:
            return x[:, :self.n_tokens_out]
        x = self.input_nn(x) if self.input_nn is not None else x

        x = x.transpose(0, 1) * math.sqrt(self.n_internal_dims)
        x = self.pos_encoder(x)
        x = self.global_layer_norm(x)
        batch_size = x.size(1)
        transformer_tgt = None
        if self.n_decoders > 0:
            transformer_tgt = self.decoder_query.unsqueeze(0).expand(batch_size, *self.decoder_query.size())
            transformer_tgt = transformer_tgt.transpose(0, 1) #* math.sqrt(self.n_internal_dims)
            # TODO: do we need tgt_norm?
        x, _ = self.transformer(x, transformer_tgt)
        x = x[:self.n_tokens_out]
        x = x.transpose(0, 1)

        x = self.output_nn(x) if self.output_nn is not None else x
        assert x.size(1) == self.n_tokens_out and x.size(2) == self.n_channels_out
        return x


class TransformerEnsembleFeaturizer(nn.Module):
    def __init__(self, ensemble_config: Dict[str, Dict[str, object]],
                 n_tokens_out, n_channels_out,
                 n_internal_dims, n_encoders, n_decoders,
                 gaussian_noise=0.0, dropout=0.0, attention_drop_proba=0.0):
        super(TransformerEnsembleFeaturizer, self).__init__()
        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        self.n_decoders = n_decoders
        if n_decoders > 0:
            decoder_query = nn.Parameter(torch.randn((n_tokens_out, n_internal_dims)) * (1 / n_internal_dims),
                                              requires_grad=True)
            self.register_parameter("decoder_query", decoder_query)

        self.ensemble_config = ensemble_config
        self.n_tokens_out = n_tokens_out
        self.n_channels_out = n_channels_out
        self.n_internal_dims = n_internal_dims
        ensemble_inp = dict()
        ensemble_id = dict()
        layer_norms = dict()
        # n_tokens_in n_channels_in is2d
        for i, (k, v) in enumerate(ensemble_config.items()):
            is2d, n_tokens_in, n_channels_in = v["is2d"], v["n_tokens_in"], v["n_channels_in"]
            # input_nn, embedding, position,
            if is2d:
                input_nn1 = nn.Conv2d(n_channels_in, n_internal_dims * 2, 1, groups=8)
                init_fc(input_nn1, "leaky_relu")
                input_nn2 = nn.Conv2d(n_internal_dims * 2, n_internal_dims, 1, groups=1)
                init_fc(input_nn2, "linear")
                input_nn = nn.Sequential(dp, input_nn1, nn.LeakyReLU(), gn, input_nn2, )

            else:
                input_nn1 = nn.Linear(n_channels_in, n_internal_dims * 2)
                init_fc(input_nn1, "leaky_relu")
                input_nn2 = nn.Linear(n_internal_dims * 2, n_internal_dims)
                init_fc(input_nn2, "linear")
                input_nn = nn.Sequential(dp, input_nn1, nn.LeakyReLU(), gn, input_nn2)
            layer_norms[k] = nn.LayerNorm(n_internal_dims)
            ensemble_inp[k] = input_nn
            ensemble_id[k] = torch.tensor(i).long().to(get_device())
        self.ensemble_inp = nn.ModuleDict(ensemble_inp)
        self.ensemble_id = ensemble_id
        self.layer_norms = nn.ModuleDict(layer_norms)

        self.output_nn = None
        if n_internal_dims != n_channels_out:
            self.output_nn = nn.Linear(n_internal_dims, n_channels_out, bias=False)
            init_fc(self.output_nn, "linear")

        self.transformer = Transformer(n_internal_dims, 8, n_encoders, n_decoders, n_internal_dims * 4, dropout, gaussian_noise, attention_drop_proba)
        self.n_tokens_in = sum([v["n_tokens_in"] for k, v in ensemble_config.items()])

        self.pos_encoder = PositionalEncoding(n_internal_dims, dropout)
        self.pos_encoder2d = PositionalEncoding2D(n_internal_dims, dropout)
        self.global_layer_norm = nn.LayerNorm(self.n_internal_dims)

    def forward(self, idict: Dict[str, torch.Tensor]):
        vecs = []
        for k, v in idict.items():
            conf = self.ensemble_config[k]
            sys.stdout.flush()
            v = self.ensemble_inp[k](v)
            if conf["is2d"]:
                v = v.transpose(1, 2).transpose(2, 3)  # B, H, W, C
                v = self.layer_norms[k](v) # R
                v = self.pos_encoder2d(v * math.sqrt(self.n_internal_dims))
            else:
                v = self.layer_norms[k](v) # R
                v = self.pos_encoder(v.transpose(0, 1) * math.sqrt(self.n_internal_dims))
            vecs.append(v)
        x = torch.cat(vecs, 0)
        assert x.size(0) == self.n_tokens_in
        x = self.global_layer_norm(x)
        batch_size = x.size(1)
        transformer_tgt = None
        if self.n_decoders > 0:
            transformer_tgt = self.decoder_query.unsqueeze(0).expand(batch_size, *self.decoder_query.size())
            transformer_tgt = transformer_tgt.transpose(0, 1) # * math.sqrt(self.n_internal_dims)
        x, _ = self.transformer(x, transformer_tgt)
        x = x[:self.n_tokens_out]
        x = x.transpose(0, 1)
        x = self.output_nn(x) if self.output_nn is not None else x
        assert x.size(1) == self.n_tokens_out and x.size(2) == self.n_channels_out
        return x



