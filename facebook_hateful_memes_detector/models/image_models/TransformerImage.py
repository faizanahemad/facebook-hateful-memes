import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from ...utils.sample import SampleList, Sample

from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, BasicFeaturizer, TransformerFeaturizer
from ..text_models import AlbertClassifer
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel, LongformerTokenizer, LongformerModel, DistilBertTokenizer, DistilBertModel
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
import torchvision.models as models
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb
from ...utils import get_device, GaussianNoise, random_word_mask, load_stored_params, ExpandContract, Transformer, PositionalEncoding, LambdaLayer, get_global, \
    get_torchvision_classification_models, get_image_info_fn, LambdaLayer, get_vgg_face_model, PositionalEncoding2D, Transpose, init_fc, dict2sampleList, \
    clean_memory, get_regularization_layers, WordMasking
from ..external.detr import get_detr_model, DETRShim
import transformers
import os
import random
import math
from ...utils.ImageModelShims import ImageCaptioningShim, ImageModelShim


class TransformerImageModel(AlbertClassifer):
    def __init__(self, image_models, classifier_dims, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=64, n_tokens_out=16,
                 head_masks=0,
                 use_as_super=False, **kwargs):
        embedding_dims = 768
        super(TransformerImageModel, self).__init__(classifier_dims, num_classes, gaussian_noise, dropout,
                                                    internal_dims, n_layers,
                                                    featurizer, final_layer_builder,
                                                    n_tokens_in, n_tokens_out, True, **kwargs)
        assert n_tokens_in % n_tokens_out == 0
        #
        self.head_masks = head_masks
        assert self.head_masks <= 12
        attention_drop_proba = kwargs.pop("attention_drop_proba", 0.0)
        self.attention_drop_proba = attention_drop_proba

        names, im_models, im_shapes, im_procs = [], [], [], []
        for imo in image_models:
            if type(imo) == dict:
                module_gaussian = imo.pop("gaussian_noise", 0.0)
                module_dropout = imo.pop("dropout", 0.0)
                imod = imo["model"]
                stored_model = imo.pop("stored_model", None)
            elif type(imo) == str:
                module_gaussian = 0.0
                module_dropout = 0.0
                imod = imo
                stored_model = None
            else:
                raise NotImplementedError()

            if "torchvision" in imod:
                im_model = ImageModelShim(resnet="_".join(imod.split("_")[1:]), dropout=module_dropout, gaussian_noise=module_gaussian, stored_model=stored_model)
                im_shape = (768, 64)
                im_proc = nn.Identity()

            elif imo == "caption_features":
                im_model = ImageCaptioningShim(module_dropout, stored_model=stored_model)
                im_shape = (768, 100)
                im_proc = nn.Identity()

            elif "detr" in imo:
                im_shape = (256, 128)
                im_model = DETRShim(128, 1, module_dropout, attention_drop_proba, device=get_device(), out_dims=embedding_dims)
                im_proc = nn.Identity()

            else:
                raise NotImplementedError(imo)

            names.append(imo)
            im_models.append(im_model)
            im_shapes.append(im_shape)
            im_procs.append(im_proc)
        self.im_models = nn.ModuleDict(dict(zip(names, im_models)))
        self.post_procs = nn.ModuleDict(dict(zip(names, im_procs)))
        self.im_shapes = dict(zip(names, im_shapes))
        self.require_raw_img = {"detr", "detr_demo", 'detr_resnet50', 'detr_resnet50_panoptic', 'detr_resnet101', 'detr_resnet101_panoptic',
                                "ssd", "faster_rcnn", "lxmert_faster_rcnn", "caption_features"}

        self.total_tokens = n_tokens_in + 1 + ((8 * int(self.n_tokens_in/(8*1.375) + 1)) if self.need_fasttext else 0) + sum([s[-1] for s in im_shapes])
        self.text_tokens = n_tokens_in

        if not use_as_super:
            model = kwargs["model"] if "model" in kwargs else 'albert-base-v2'
            model_class = AutoModel
            tokenizer_class = AutoTokenizer

            global_dir = get_global("models_dir")
            model = os.path.join(global_dir, model) if model in os.listdir(global_dir) else model
            self.tokenizer = tokenizer_class.from_pretrained(model)
            self.model = model_class.from_pretrained(model)
            print("Pick stored Model", model, "Model Class = ", type(self.model), "Tokenizer Class = ", type(self.tokenizer))
            if featurizer == "transformer":
                n_encoders = kwargs.pop("n_encoders", n_layers)
                n_decoders = kwargs.pop("n_decoders", n_layers)
                self.featurizer = TransformerFeaturizer(self.total_tokens, embedding_dims, n_tokens_out,
                                                        classifier_dims,
                                                        internal_dims, n_encoders, n_decoders,
                                                        gaussian_noise, dropout, self.attention_drop_proba)
            else:
                raise NotImplementedError()

            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)

        self.LayerNorm = nn.LayerNorm(embedding_dims, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.word_masking = WordMasking(tokenizer=self.tokenizer, **kwargs)
        self.reg_layers = get_regularization_layers(self)

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.text_tokens
        texts = self.word_masking(texts)
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def get_vectors(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        img = sampleList.torchvision_image
        image = sampleList.image
        mixup = sampleList.mixup
        input_ids, attention_mask = self.tokenise(sampleList.text)
        word_embeddings = self.model.embeddings(input_ids) # B, S, C
        image_vectors = list()
        if len(set(self.im_models.keys()) - self.require_raw_img) > 0:
            img = img.to(get_device())

        if self.need_fasttext:
            fasttext_vectors = self.fasttext_vectors(sampleList.text)
            seq_length = word_embeddings.size(1)
            position_ids = torch.arange(seq_length + 1, seq_length + fasttext_vectors.size(1), dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
            position_embeddings = self.model.embeddings.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
            fasttext_vectors = fasttext_vectors + position_embeddings  # (bs, max_seq_length, dim)
            fasttext_vectors = self.LayerNorm(fasttext_vectors)  # (bs, max_seq_length, dim)
            fasttext_vectors = self.dropout(fasttext_vectors)  # (bs, max_seq_length, dim)
            attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0), fasttext_vectors.size(1))], 1)
            word_embeddings = torch.cat([word_embeddings, fasttext_vectors], 1)

        for k, m in self.im_models.items():
            x = image if k in self.require_raw_img else img
            kw = dict(ignore_cache=mixup) if k in self.require_raw_img else dict()
            im_repr = m(x, **kw)
            im_repr = self.post_procs[k](im_repr)
            image_vectors.append(im_repr.to(get_device()))
            clean_memory()

        image_vectors = torch.cat(image_vectors, 1)
        seq_length = word_embeddings.size(1)
        position_ids = torch.arange(seq_length, seq_length + image_vectors.size(1), dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand(image_vectors.size()[:2])  # (bs, max_seq_length)
        position_embeddings = self.model.embeddings.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        image_vectors = image_vectors * math.sqrt(image_vectors.size(-1)) + position_embeddings  # (bs, max_seq_length, dim)
        image_vectors = self.LayerNorm(image_vectors)  # (bs, max_seq_length, dim)
        image_vectors = self.dropout(image_vectors)  # (bs, max_seq_length, dim)
        attention_mask = attention_mask.to(get_device())
        image_vectors = image_vectors.to(get_device())
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0), image_vectors.size(1), device=get_device(), dtype=attention_mask.dtype)], 1)
        embeddings = torch.cat([word_embeddings, image_vectors], 1)

        if self.training:
            head_mask = [1] * (12 - self.head_masks) + [0] * self.head_masks
            random.shuffle(head_mask)
        else:
            head_mask = [1] * 12
        encoder = getattr(self.model, "transformer", getattr(self.model, "encoder", None))
        if type(self.model) == transformers.modeling_longformer.LongformerModel:
            attention_window = (
                self.model.config.attention_window
                if isinstance(self.model.config.attention_window, int)
                else max(self.model.config.attention_window)
            )
            padding_len, input_ids, attention_mask, token_type_ids, position_ids, embeddings = self.model._pad_to_window_size(
                input_ids=None,
                attention_mask=attention_mask,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=embeddings,
                attention_window=attention_window,
                pad_token_id=self.model.config.pad_token_id,
            )
            attention_mask = attention_mask.unsqueeze(2)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        tfmr_output = encoder(embeddings, attention_mask, head_mask=head_mask)
        hidden_state = tfmr_output[0]
        return (hidden_state,)

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        labels = torch.tensor(sampleList.label).to(get_device())
        # sample_weights = torch.tensor(sampleList.sample_weight, dtype=float).to(get_device())
        vectors = self.get_vectors(sampleList)[-1]
        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels) if self.final_layer is not None else (None, None)

        if self.training:
            loss += self.auc_dice_loss(logits, labels)
        return logits, vectors.mean(1), vectors, loss
