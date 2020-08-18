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
    clean_memory
from ..external.detr import get_detr_model, DETRShim
import transformers
import os
import random
import math
from ...utils.ImageModelShims import ImageCaptioningShim, ImageModelShim, ImageModelShimSimple

# t5-small albert-base-v2 ctrl distilgpt2 google/electra-base-generator microsoft/DialoGPT-small allenai/scibert_scivocab_uncased activebus/BERT_Review allenai/reviews_roberta_base
# Hate-speech-CNERG/dehatebert-mono-english nlptown/bert-base-multilingual-uncased-sentiment

#

# allenai/scibert_scivocab_uncased allenai/biomed_roberta_base distilgpt2 google/electra-base-generator ctrl t5-small
# mrm8488/t5-base-finetuned-math-qa-test malteos/arqmath-bert-base-cased

# Keep per head loss stats, make GRU heads separate

class TransformerImageV2Model(AlbertClassifer):
    def __init__(self, image_model, classifier_dims, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, n_layers,
                 featurizer, final_layer_builder,
                 n_tokens_in=64, n_tokens_out=16,
                 head_masks=0,
                 use_as_super=False, **kwargs):
        embedding_dims = 768
        super(TransformerImageV2Model, self).__init__(classifier_dims, num_classes, gaussian_noise, dropout,
                                                      internal_dims, n_layers,
                                                      featurizer, final_layer_builder,
                                                      n_tokens_in, n_tokens_out, True, **kwargs)
        assert n_tokens_in % n_tokens_out == 0
        #
        self.head_masks = head_masks
        assert self.head_masks <= 12
        attention_drop_proba = kwargs.pop("attention_drop_proba", 0.0)
        self.attention_drop_proba = attention_drop_proba

        if type(image_model) == dict:
            module_gaussian = image_model.pop("gaussian_noise", 0.0)
            module_dropout = image_model.pop("dropout", 0.0)
            stored_model = image_model.pop("stored_model", None)
            im_model = ImageModelShimSimple(resnet="resnet18_swsl", dropout=module_dropout, gaussian_noise=module_gaussian, stored_model=stored_model)
        elif type(image_model) == str:
            module_gaussian = 0.0
            module_dropout = 0.0
            stored_model = image_model
            im_model = ImageModelShimSimple(resnet="resnet18_swsl", dropout=module_dropout, gaussian_noise=module_gaussian, stored_model=stored_model)
        elif type(image_model) == ImageModelShimSimple:
            im_model = image_model
        else:
            raise NotImplementedError()

        im_shape = (768, 64)
        im_proc = nn.Identity()
        self.im_model = im_model
        self.post_proc = im_proc
        self.im_shape = im_shape

        self.total_tokens = n_tokens_in + 1
        self.text_tokens = n_tokens_in

        model = kwargs["model"]
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

        self.reg_layers = [(c, c.p if hasattr(c, "p") else c.sigma) for c in self.children() if c.__class__ == GaussianNoise or c.__class__ == nn.Dropout]

    def tokenise(self, texts: List[str]):
        tokenizer = self.tokenizer
        n_tokens_in = self.text_tokens
        if self.training and self.word_masking_proba > 0:
            texts = [random_word_mask(t, tokenizer, self.word_masking_proba) for t in texts]
        converted_texts = tokenizer.batch_encode_plus(texts, add_special_tokens=True, pad_to_max_length=True, max_length=n_tokens_in, truncation=True)
        input_ids, attention_mask = converted_texts["input_ids"], converted_texts["attention_mask"]
        return torch.tensor(input_ids).to(get_device()), torch.tensor(attention_mask).to(get_device())

    def get_vectors(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        input_ids, attention_mask = self.tokenise(sampleList.text)
        word_embeddings = self.model.embeddings(input_ids)  # B, S, C
        if hasattr(sampleList, "torchvision_image"):
            img = sampleList.torchvision_image
            img = img.to(get_device())
            im_repr = self.im_model(img)
            im_repr = self.post_proc(im_repr).to(get_device())
            image_vectors = im_repr[:, 0].unsqueeze(1)
            clean_memory()
        else:
            image_vectors = torch.zeros(word_embeddings.size(0), 1, word_embeddings.size(2), dtype=torch.float, device=get_device())
        seq_length = word_embeddings.size(1)
        position_ids = torch.arange(seq_length, seq_length + image_vectors.size(1), dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand(image_vectors.size()[:2])  # (bs, max_seq_length)
        position_embeddings = self.model.embeddings.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        image_vectors = image_vectors + position_embeddings  # (bs, max_seq_length, dim)
        image_vectors = self.LayerNorm(image_vectors)  # (bs, max_seq_length, dim)
        image_vectors = self.dropout(image_vectors)  # (bs, max_seq_length, dim)
        attention_mask = attention_mask.to(get_device())
        image_vectors = image_vectors.to(get_device())
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0), image_vectors.size(1), device=get_device(), dtype=attention_mask.dtype)],
                                   1)
        embeddings = torch.cat([word_embeddings, image_vectors], 1)

        if self.training:
            head_mask = [1] * (12 - self.head_masks) + [0] * self.head_masks
            random.shuffle(head_mask)
        else:
            head_mask = [1] * 12
        encoder = getattr(self.model, "transformer", getattr(self.model, "encoder", None))

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        tfmr_output = encoder(embeddings, attention_mask, head_mask=head_mask)
        hidden_state = tfmr_output[0]
        output = (hidden_state,) + tfmr_output[1:]
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
