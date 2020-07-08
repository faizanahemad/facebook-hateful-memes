from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
from .BaseFeaturizer import BaseFeaturizer
from ...utils import init_fc, GaussianNoise, ExpandContract, LambdaLayer, get_device
import math
from .CNN1DFeaturizer import Residual1DConv
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import sys
import copy
from typing import Optional, Any
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, LayerNorm, TransformerEncoderLayer


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

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        assert x.size(-1) == self.d_model
        x = x.transpose(0, 1).transpose(1, 2)  # H, W, B, C
        pe = self.pe[:x.size(0), :] # H, C
        pe_abs = self.pe[:x.size(0) * x.size(1), :]
        # print("2D PE", pe.size(), self.pe.size(), torch.max(pe))
        # pe = pe.expand(pe.size(0), b, self.d_model) # H, B, C
        pe1 = pe.unsqueeze(1)
        pe2 = pe.unsqueeze(0)
        # print("2D PE", x.size(), pe.size(), pe1.size(), pe2.size())
        # sys.stdout.flush()
        x = x + (pe1 + pe2)/3
        x = x.flatten(0, 1) + pe_abs/3
        return self.dropout(x)


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
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu") -> None:
        super(Transformer, self).__init__()
        assert num_encoder_layers > 0 or num_decoder_layers > 0
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if num_decoder_layers > 0:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

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


class TransformerFeaturizer(nn.Module):
    def __init__(self, n_tokens_in, n_channels_in, n_tokens_out, n_channels_out,
                 n_internal_dims, n_encoders, n_decoders,
                 gaussian_noise=0.0, dropout=0.0):
        super(TransformerFeaturizer, self).__init__()
        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        self.n_tokens_out = n_tokens_out
        self.n_channels_out = n_channels_out
        self.n_internal_dims = n_internal_dims
        self.n_decoders = n_decoders
        if n_decoders > 0:
            decoder_query = nn.Parameter(torch.randn((n_tokens_out, n_internal_dims)) * (1 / n_internal_dims),
                                         requires_grad=True)
            self.register_parameter("decoder_query", decoder_query)
            self.tgt_norm = nn.LayerNorm(n_internal_dims)

        self.input_nn = None
        if n_channels_in != n_internal_dims:
            self.input_nn = nn.Linear(n_channels_in, n_internal_dims, bias=False)
            init_fc(self.input_nn, "linear")

        self.output_nn = None
        if n_internal_dims != n_channels_out:
            self.output_nn = nn.Linear(n_internal_dims, n_channels_out, bias=False)
            init_fc(self.output_nn, "linear")

        self.transformer = Transformer(n_internal_dims, 16, n_encoders, n_decoders, n_internal_dims*4, dropout)
        self.pos_encoder = PositionalEncoding(n_internal_dims, dropout)
        self.global_layer_norm = nn.LayerNorm(n_internal_dims)

    def forward(self, x):
        x = self.input_nn(x) if self.input_nn is not None else x

        x = x.transpose(0, 1) * math.sqrt(self.n_internal_dims)
        x = self.pos_encoder(x)
        x = self.global_layer_norm(x)
        batch_size = x.size(1)
        transformer_tgt = None
        if self.n_decoders > 0:
            transformer_tgt = self.decoder_query.unsqueeze(0).expand(batch_size, *self.decoder_query.size())
            transformer_tgt = transformer_tgt.transpose(0, 1) * math.sqrt(self.n_internal_dims)
            transformer_tgt = self.pos_encoder(transformer_tgt)
            transformer_tgt = self.tgt_norm(transformer_tgt)
        x, m = self.transformer(x, transformer_tgt)
        m = m[:self.n_tokens_out]
        x = x[:self.n_tokens_out]
        x = x.transpose(0, 1)

        x = self.output_nn(x) if self.output_nn is not None else x
        assert x.size(1) == self.n_tokens_out and x.size(2) == self.n_channels_out
        return x


class TransformerEnsembleFeaturizer(nn.Module):
    def __init__(self, ensemble_config: Dict[str, Dict[str, object]],
                 n_tokens_out, n_channels_out,
                 n_internal_dims, n_encoders, n_decoders,
                 gaussian_noise=0.0, dropout=0.0):
        super(TransformerEnsembleFeaturizer, self).__init__()
        gn = GaussianNoise(gaussian_noise)
        dp = nn.Dropout(dropout)
        self.n_decoders = n_decoders
        if n_decoders > 0:
            decoder_query = nn.Parameter(torch.randn((n_tokens_out, n_internal_dims)) * (1 / n_internal_dims),
                                              requires_grad=True)
            self.register_parameter("decoder_query", decoder_query)
            self.tgt_norm = nn.LayerNorm(n_internal_dims)

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

        self.transformer = Transformer(n_internal_dims, 16, n_encoders, n_decoders, n_internal_dims * 4, dropout)
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
                v = self.layer_norms[k](v)
                v = self.pos_encoder2d(v * math.sqrt(self.n_internal_dims))
            else:
                v = self.layer_norms[k](v)
                v = self.pos_encoder(v.transpose(0, 1) * math.sqrt(self.n_internal_dims))
            vecs.append(v)
        x = torch.cat(vecs, 0)
        assert x.size(0) == self.n_tokens_in
        x = self.global_layer_norm(x)
        batch_size = x.size(1)
        transformer_tgt = None
        if self.n_decoders > 0:
            transformer_tgt = self.decoder_query.unsqueeze(0).expand(batch_size, *self.decoder_query.size())
            transformer_tgt = transformer_tgt.transpose(0, 1) * math.sqrt(self.n_internal_dims)
            transformer_tgt = self.pos_encoder(transformer_tgt)
            transformer_tgt = self.tgt_norm(transformer_tgt)
        x, m = self.transformer(x, transformer_tgt)
        x = x[:self.n_tokens_out]
        m = m[:self.n_tokens_out]
        x = x.transpose(0, 1)
        x = self.output_nn(x) if self.output_nn is not None else x
        assert x.size(1) == self.n_tokens_out and x.size(2) == self.n_channels_out
        return x



