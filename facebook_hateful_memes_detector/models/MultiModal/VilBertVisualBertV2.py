import operator
import os
import time
from collections import defaultdict

import torchvision
import torchvision.models as models

import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union, Callable
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
import fasttext
from sklearn.metrics import accuracy_score
from torch.utils.checkpoint import checkpoint

from ...utils.sample import SampleList, Sample
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...training import get_auc_dice_loss
from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_torchvision_classification_models, get_device, get_image_info_fn, Transpose, \
    dict2sampleList, loss_calculator, get_loss_by_task, clean_memory, pad_tensor, random_word_mask, load_stored_params, LinearHead, get_regularization_layers, \
    WordMasking, FeatureDropout, MLMPretraining, BertLMPredictionHead
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel
from torch.utils.checkpoint import checkpoint

from ..external.lxrt import get_lxrt_model
import GPUtil
import random

# TODO: From each of Vilbert/visual_bert/LXMERT/MMBT take only 32 first tokens and then 32 tokens after 96 = 64 x 4 = 256 tokens/seq
# For These 256 Seq Do a Self-attn encoder layers before decoder-ensemble head. Pretrain the self-attn decoder keeping backbones const.
# Pretrain all backbones before doing combo backbones.


def identity(x): return x


class VilBertVisualBertModelV2(nn.Module):
    def __init__(self, model_name: Union[List, Dict], num_classes,
                 gaussian_noise, dropout, classifier_dims,
                 featurizer,
                 n_tokens_in,
                 loss,
                 **kwargs):
        super(VilBertVisualBertModelV2, self).__init__()
        from ..external.mmf import get_vilbert, get_visual_bert, get_tokenizer, get_mmbt_region
        self.word_masking_proba = kwargs["word_masking_proba"] if "word_masking_proba" in kwargs else 0.0
        max_seq_length = n_tokens_in
        self.text_tokens = max_seq_length
        assert type(loss) == str and loss in ["classification", "focal", "regression", "k-classification"]
        self.task = loss
        assert max_seq_length >= 64
        self.label_not_present = -1
        self.max_seq_length = max_seq_length
        self.text_processor = get_tokenizer(max_seq_length)
        n_tokens_in, pooled_dims = 0, 0
        model_name = [model_name] if type(model_name) == str else model_name
        self.model_regularizers = nn.ModuleDict()
        self.model_heads = nn.ModuleDict()
        self.bbox_swaps = kwargs.pop("bbox_swaps", 0)
        self.bbox_copies = kwargs.pop("bbox_copies", 0)
        self.bbox_deletes = kwargs.pop("bbox_deletes", 0)
        self.bbox_gaussian_noise = GaussianNoise(kwargs.pop("bbox_gaussian_noise", 0.0))
        self.bbox_dropout = nn.Dropout(kwargs.pop("bbox_dropout", 0.0))
        self.bbox_feature_dropout = FeatureDropout(kwargs.pop("bbox_feature_dropout", 0.0))
        self.view_transforms = kwargs.pop("view_transforms", list())
        self.view_loss_weight = kwargs.pop("view_loss_weight", 0.1)

        self.full_loss_hist = list()
        self.view_loss_hist = list()

        self.vilbert_accuracy_hist = list()
        self.mmbt_region_accuracy_hist = list()
        self.visual_bert_accuracy_hist = list()
        self.lxmert_accuracy_hist = list()

        self.full_accuracy_hist = list()
        self.logit_accuracy_hist = list()
        self.pre_logit_accuracy_hist = list()
        self.pooled_logit_accuracy_hist = list()
        assert type(model_name) == dict
        for k, v in model_name.items():
            dp = nn.Dropout(v["dropout"] if "dropout" in v else 0.0)
            fdp = FeatureDropout(v["feature_dropout"] if "feature_dropout" in v else 0.0)
            gn = GaussianNoise(v["gaussian_noise"] if "gaussian_noise" in v else 0.0)
            self.model_regularizers[k] = nn.Sequential(fdp, gn, dp)

        self.model_name = model_name
        self.vilbert = get_vilbert(get_device())
        n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + 100 + max_seq_length, 768, pooled_dims + 1024
        for p in self.vilbert.parameters():
            p.requires_grad = False
        lin = nn.Linear(1024, num_classes)
        init_fc(lin, "linear")
        self.model_heads["vilbert"] = lin

        self.visual_bert = get_visual_bert(get_device())
        n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + 100 + max_seq_length, 768, pooled_dims + 768
        for p in self.visual_bert.parameters():
            p.requires_grad = False
        lin = nn.Linear(768, num_classes)
        init_fc(lin, "linear")
        self.model_heads["visual_bert"] = lin

        self.lxmert = get_lxrt_model("20", pretokenized=True, max_seq_len=max_seq_length)
        n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + max_seq_length + 36, 768, pooled_dims + 768
        self.lxmert.to(get_device())
        for p in self.lxmert.parameters():
            p.requires_grad = False
        lin = nn.Linear(768, num_classes)
        init_fc(lin, "linear")
        self.model_heads["lxmert"] = lin


        self.mmbt_region = get_mmbt_region(get_device())
        n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + 102 + max_seq_length, 768, pooled_dims + 768
        for p in self.mmbt_region.parameters():
            p.requires_grad = False
        lin = nn.Linear(768, num_classes)
        init_fc(lin, "linear")
        self.model_heads["mmbt_region"] = lin

        if len(set(model_name.keys()) - {"vilbert", "visual_bert", "lxmert", "mmbt_region"}) > 0:
            raise NotImplementedError()

        self.get_img_details = get_image_info_fn(enable_encoder_feats=False, device=get_device())["get_img_details"]
        self.get_lxmert_details = get_image_info_fn(enable_encoder_feats=False, device=get_device())["get_lxmert_details"]
        self.n_tokens_out = max_seq_length + 36
        print("N tokens Out = ", self.n_tokens_out, "Classifier Dims = ", classifier_dims, "Matches embedding_dims: ", embedding_dims == classifier_dims)

        self.featurizer_type = featurizer

        self.num_classes = num_classes
        lin0 = nn.Linear(pooled_dims, 512)
        init_fc(lin0, "leaky_relu")
        lin = nn.Linear(512, num_classes)
        init_fc(lin, "linear")
        dp = nn.Dropout(dropout)
        self.one_view_layer = nn.Sequential(dp, lin0, nn.LeakyReLU(), lin)

        self.pooled_dims = pooled_dims * (len(self.view_transforms) + 1)
        lin0 = nn.Linear(self.pooled_dims, pooled_dims)
        init_fc(lin0, "leaky_relu")
        lin01 = nn.Linear(pooled_dims, pooled_dims)
        init_fc(lin01, "leaky_relu")
        lin1 = nn.Linear(pooled_dims, 512)
        init_fc(lin1, "leaky_relu")
        lin = nn.Linear(512, num_classes)
        init_fc(lin, "linear")
        dp = nn.Dropout(dropout)
        self.final_layer = nn.Sequential(lin0, nn.LeakyReLU(), GaussianNoise(gaussian_noise),
                                         lin01, nn.LeakyReLU(), dp,
                                         lin1, nn.LeakyReLU(), lin)
        uda = kwargs.pop("uda", False)
        self.loss = get_loss_by_task(loss, num_classes if uda else None)
        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.word_masking = WordMasking(tokenizer=self.text_processor._tokenizer, **kwargs)
        self.reg_layers = get_regularization_layers(self)
        self.auc_loss_coef = kwargs.pop("auc_loss_coef", 0.0)
        self.dice_loss_coef = kwargs.pop("dice_loss_coef", 0.0)
        self.auc_method = kwargs.pop("auc_method", 1)
        self.auc_dice_loss = get_auc_dice_loss(num_classes, self.dice_loss_coef, self.auc_loss_coef, auc_method=self.auc_method)

    def get_tokens(self, texts):
        keys = ["input_ids", "input_mask", "segment_ids"]
        texts = self.word_masking(texts)
        texts = [self.text_processor({"text": t}) for t in texts]
        texts = SampleList([Sample({k: t[k] for k in keys}) for t in texts])
        for k in keys:
            texts[k] = texts[k].to(get_device())
        return texts

    def build_lxmert_sample_list(self, orig_image, textSampleList: SampleList):
        imgfs = [self.get_lxmert_details(im, ignore_cache=False) for im, ignore_cache in orig_image]
        samples = [Sample(dict(feats=pad_tensor(feats, 36),
                               boxes=pad_tensor(boxes.pred_boxes.tensor, 36),
                               masks=torch.tensor(([1] * len(feats)) + ([0] * (36 - len(feats)))).long())) for boxes, feats in imgfs]
        samples = [self.bbox_aug(s, "lxmert") for s in samples]
        sl = SampleList(samples)
        sl.input_ids = textSampleList.input_ids
        sl.input_mask = textSampleList.input_mask
        sl.segment_ids = textSampleList.segment_ids
        return sl

    def bbox_aug(self, sample,
                 extractor_type="vilbert_visual_bert"):

        # TODO: Do manual inspection
        if not self.training:
            return sample
        swaps = self.bbox_swaps
        copies = self.bbox_copies
        deletes = self.bbox_deletes
        if extractor_type == "vilbert_visual_bert":
            imf = self.bbox_gaussian_noise(sample["image_feature_0"])
            imf = self.bbox_dropout(imf)
            imf = self.bbox_feature_dropout(imf)
            imi = sample["image_info_0"]
            bbox = self.bbox_gaussian_noise(imi["bbox"])
            cls_prob = self.bbox_gaussian_noise(imi["cls_prob"])
            changes = [imf, bbox, cls_prob]

            for i in range(swaps):
                swap = random.sample(range(100), 2)
                for v in changes:
                    t = v[swap[1]]
                    v[swap[1]] = v[swap[0]]
                    v[swap[0]] = t

            for i in range(copies):
                copi = random.sample(range(100), 2)
                for v in changes:
                    v[copi[0]] = v[copi[1]]

            for i in range(deletes):
                copi = random.sample(range(99), 2)
                for v in changes:
                    v[copi[0]] = v[copi[1]]

        elif extractor_type == "lxmert":
            for k, v in sample.items():
                if k == "feats" or k == "boxes":
                    v = self.bbox_gaussian_noise(v)
                    if k == "feats":
                        v = self.bbox_dropout(v)
                        v = self.bbox_feature_dropout(v)
                    sample[k] = v
            for i in range(swaps):
                swap = random.sample(range(36), 2)
                for k, v in sample.items():
                    t = v[swap[1]]
                    v[swap[1]] = v[swap[0]]
                    v[swap[0]] = t
                    sample[k] = v

            for i in range(copies):
                copi = random.sample(range(36), 2)
                for k, v in sample.items():
                    v[copi[0]] = v[copi[1]]

            for i in range(deletes):
                copi = random.sample(range(36), 2)
                for k, v in sample.items():
                    v[copi[0]] = v[copi[1]]

        else:
            raise NotImplementedError
        return sample

    def build_vilbert_visual_bert_sample_list(self, orig_image, textSampleList: SampleList, mixup: List[bool]):
        # Rank swap higher and lower ranked boxes+features
        # Copy one bbox to another and erase the 2nd one entirely
        imgfs = [self.get_img_details(im, ignore_cache=ignore_cache) for im, ignore_cache in zip(orig_image, mixup)]
        samples = [Sample(dict(image_feature_0=feat_list, image_info_0=info_list)) for feat_list, info_list in imgfs]
        samples = [self.bbox_aug(s, "vilbert_visual_bert") for s in samples]
        sl = SampleList(samples)
        sl.input_ids = textSampleList.input_ids
        sl.input_mask = textSampleList.input_mask
        sl.segment_ids = textSampleList.segment_ids
        sl.id = textSampleList.id
        return sl

    def __vilbert_preprocessing__(self, sample_list: SampleList):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        image_info = getattr(sample_list, "image_info_0", {})
        image_dim_variable = torch.tensor(getattr(image_info, "max_features", None))
        image_dim_variable = image_dim_variable.to(get_device())
        image_feature_variable = getattr(sample_list, "image_feature_0", None)
        image_feature_variable = image_feature_variable.to(get_device())
        image_label_variable = getattr(sample_list, "image_labels", None)
        if image_label_variable is not None:
            image_label_variable = torch.tensor(
                image_label_variable, dtype=torch.long
            )

        bbox = np.array(getattr(image_info, "bbox", None), dtype=np.float32)
        image_w = np.array(
            getattr(image_info, "image_width", None), dtype=np.float32
        )
        image_h = np.array(
            getattr(image_info, "image_height", None), dtype=np.float32
        )
        image_location = np.zeros(
            (bbox.shape[0], bbox.shape[1], 5), dtype=np.float32
        )
        image_location[:, :, :4] = bbox
        image_location[:, :, 4] = (
                (image_location[:, :, 3] - image_location[:, :, 1])
                * (image_location[:, :, 2] - image_location[:, :, 0])
                / (image_w * image_h)[:, None]
        )
        image_location[:, :, 0] = image_location[:, :, 0] / image_w[:, None]
        image_location[:, :, 1] = image_location[:, :, 1] / image_h[:, None]
        image_location[:, :, 2] = image_location[:, :, 2] / image_w[:, None]
        image_location[:, :, 3] = image_location[:, :, 3] / image_h[:, None]
        image_location_variable = torch.tensor(
            image_location, dtype=torch.float, device=get_device()
        )

        cls_prob = getattr(image_info, "cls_prob", None)
        image_target = np.array(cls_prob, dtype=np.float32)
        image_target_variable = torch.tensor(image_target, dtype=torch.float)
        image_feature_variable = image_feature_variable.to(get_device())
        params = {"input_ids": bert_input_ids, "attention_mask": bert_input_mask, "token_type_ids": bert_input_type_ids, "image_dim": image_dim_variable,
                  "image_feature": image_feature_variable, "image_location": image_location_variable, "image_target": image_target_variable,
                  "image_label": image_label_variable, "masked_lm_labels": getattr(sample_list, "lm_label_ids", None)}


        # Prepare Mask
        if params["image_feature"] is not None and params["image_dim"] is not None:
            image_mask = (torch.arange(params["image_feature"].size(-2), device=get_device()).expand(*params["image_feature"].size()[:-1]))
            if len(params["image_dim"].size()) < len(image_mask.size()):
                params["image_dim"] = params["image_dim"].unsqueeze(-1)
                assert len(params["image_dim"].size()) == len(image_mask.size())
            image_mask = image_mask < params["image_dim"]
            params["image_attention_mask"] = image_mask.long()
        else:
            params["image_attention_mask"] = None
        params.pop("image_dim")
        params = {"input_ids": params["input_ids"], "image_feature": params["image_feature"], "image_location": params["image_location"],
                  "token_type_ids": params["token_type_ids"], "attention_mask": params["attention_mask"], "image_attention_mask": params["image_attention_mask"]}
        clean_memory()
        params = {k: v.to(get_device()) if type(v) == torch.Tensor else v for k, v in params.items()}
        return params

    def vilbert_processor(self, sample_list: SampleList):
        params = self.__vilbert_preprocessing__(sample_list)
        clean_memory()
        # GPUtil.showUtilization()

        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            attention_weights,
        ) = self.vilbert.model.bert(
            params["input_ids"],
            params["image_feature"],
            params["image_location"],
            params["token_type_ids"],
            params["attention_mask"],
            params["image_attention_mask"],
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
        )
        del params
        clean_memory()
        if self.vilbert.model.fusion_method == "sum":
            pooled_output = self.vilbert.model.dropout(pooled_output_t + pooled_output_v)
        elif self.vilbert.model.fusion_method == "mul":
            pooled_output = self.vilbert.model.dropout(pooled_output_t * pooled_output_v)
        else:
            raise AssertionError
        sequence_output_v = sequence_output_v[:, :, :sequence_output_t.size(-1)]
        seq = torch.cat([sequence_output_v, sequence_output_t], 1)
        seq = self.model_regularizers["vilbert"](seq) if "vilbert" in self.model_regularizers else seq

        logits = self.model_heads["vilbert"](pooled_output)
        logits = logits / logits.norm(dim=1, keepdim=True).clamp(min=1e-5)
        output = dict(sequence_output=seq,
                      pooled_output=pooled_output,
                      logits=logits)
        return output

    def __visual_bert_preprocessing__(self, sample_list: SampleList):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids
        image_info = getattr(sample_list, "image_info_0", {})
        image_dim_variable = torch.tensor(getattr(image_info, "max_features", None))
        image_dim_variable = image_dim_variable.to(get_device())
        image_feat_variable = getattr(sample_list, "image_feature_0", None)
        image_feat_variable = image_feat_variable.to(get_device())

        sample_list.visual_embeddings = image_feat_variable
        sample_list.image_dim = image_dim_variable
        sample_list.input_ids = bert_input_ids
        sample_list.input_mask = bert_input_mask
        sample_list.token_type_ids = bert_input_type_ids

        visual_embeddings = getattr(sample_list, "visual_embeddings", None)
        image_dim = getattr(sample_list, "image_dim", None)
        image_dim = image_dim.to(get_device())
        # pretraining labels
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None and image_dim is not None:
            image_mask = (
                torch.arange(visual_embeddings.size(-2), device=get_device())
                    .expand(*visual_embeddings.size()[:-1])
            )
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
                assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask < image_dim
            sample_list.image_mask = image_mask.long()
        else:
            sample_list.image_mask = None

        sample_list.position_embeddings_visual = None

        sample_list.image_mask = sample_list.image_mask.to(get_device())
        sample_list.input_mask = sample_list.input_mask.to(get_device())
        sample_list = self.visual_bert.flatten_for_bert(sample_list)

        sample_list.input_ids = sample_list.input_ids.to(get_device())
        sample_list.attention_mask = sample_list.attention_mask.to(get_device())
        sample_list.token_type_ids = sample_list.token_type_ids.to(get_device())
        sample_list.visual_embeddings = sample_list.visual_embeddings.to(get_device())
        sample_list.visual_embeddings_type = sample_list.visual_embeddings_type.to(get_device())
        params = {"input_ids": sample_list.input_ids, "attention_mask": sample_list.attention_mask, "token_type_ids": sample_list.token_type_ids,
                  "visual_embeddings": sample_list.visual_embeddings, "position_embeddings_visual": sample_list.position_embeddings_visual,
                  "visual_embeddings_type": sample_list.visual_embeddings_type,
                  "image_text_alignment": sample_list.image_text_alignment,}
        return params

    def visual_bert_processor(self, params: Dict):
        sequence_output, pooled_output, attention_weights = self.visual_bert.model.bert(
            params["input_ids"],
            params["attention_mask"],
            params["token_type_ids"],
            params["visual_embeddings"],
            params["position_embeddings_visual"],
            params["visual_embeddings_type"],
            params["image_text_alignment"],
        )
        output_dict = {}
        output_dict["sequence_output"] = sequence_output
        output_dict["pooled_output"] = pooled_output
        del params
        return output_dict

    def visual_bert_forward(self, sl: SampleList):
        params = self.__visual_bert_preprocessing__(sl)
        del sl
        out = self.visual_bert_processor(params)
        out["sequence_output"] = self.model_regularizers["visual_bert"](out["sequence_output"]) if "visual_bert" in self.model_regularizers else out["sequence_output"]
        out["pooled_output"] = self.model_regularizers["visual_bert"](out["pooled_output"]) if "visual_bert" in self.model_regularizers else out["pooled_output"]

        logits = self.model_heads["visual_bert"](out["pooled_output"])
        logits = logits / logits.norm(dim=1, keepdim=True).clamp(min=1e-5)
        out["logits"] = logits
        return out

    def lxmert_forward(self, orig_image, textSampleList):
        lx_sl = self.build_lxmert_sample_list(orig_image, textSampleList)
        for k, v in lx_sl.items():
            if type(v) == torch.Tensor:
                lx_sl[k] = v.to(get_device())
        feat_seq, pooled = self.lxmert((lx_sl.input_ids, lx_sl.input_mask, lx_sl.segment_ids,), (lx_sl.feats, lx_sl.boxes), lx_sl.masks)

        del lx_sl
        seq = torch.cat(feat_seq, 1)
        seq = self.model_regularizers["lxmert"](seq) if "lxmert" in self.model_regularizers else seq
        pooled = self.model_regularizers["lxmert"](pooled) if "lxmert" in self.model_regularizers else pooled
        logits = self.model_heads["lxmert"](pooled)
        logits = logits / logits.norm(dim=1, keepdim=True).clamp(min=1e-5)
        return dict(seq=seq, pooled=pooled, logits=logits)

    def mmbt_region_forward(self, sl: SampleList):
        sl = sl.to(get_device())
        sl.image_feature_0 = sl.image_feature_0.type(torch.float)
        module_output = self.mmbt_region.model.bert(sl)
        pooled_output = module_output[1]
        output = {}
        seq = self.model_regularizers["mmbt_region"](module_output[0]) if "mmbt_region" in self.model_regularizers else seq
        output["sequence_output"] = seq
        output["pooled_output"] = pooled_output
        logits = self.model_heads["mmbt_region"](pooled_output)
        logits = logits / logits.norm(dim=1, keepdim=True).clamp(min=1e-5)
        output["logits"] = logits
        del sl
        return output

    def get_vectors(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        texts = sampleList.text
        image = sampleList.image  # orig_image = sampleList.original_image
        labels = torch.tensor(sampleList.label, dtype=float).to(get_device())
        mixup = sampleList.mixup

        actual_labels = np.array(labels.tolist())
        indices = actual_labels != self.label_not_present
        actual_labels = actual_labels[indices]

        textSampleList = self.get_tokens(texts)
        textSampleList.id = sampleList.id
        del sampleList
        # GPUtil.showUtilization()
        pooled_output = []
        sequence_output = []
        logit = []

        sl = self.build_vilbert_visual_bert_sample_list(image, textSampleList, mixup)

        out = self.vilbert_processor(sl)
        seq = out["sequence_output"]
        sequence_output.append(seq)
        pool = out["pooled_output"]
        pooled_output.append(pool)
        logit.append(out["logits"])

        predicted_labels = np.array(out["logits"].max(dim=1).indices.tolist())
        predicted_labels = predicted_labels[indices]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        self.vilbert_accuracy_hist.append(accuracy)

        out = self.mmbt_region_forward(sl)
        seq, pool = out["sequence_output"], out["pooled_output"]
        logit.append(out["logits"])
        sequence_output.append(seq)
        pooled_output.append(pool)

        predicted_labels = np.array(out["logits"].max(dim=1).indices.tolist())
        predicted_labels = predicted_labels[indices]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        self.mmbt_region_accuracy_hist.append(accuracy)

        out = self.visual_bert_forward(sl)
        seq, pool = out["sequence_output"], out["pooled_output"]
        logit.append(out["logits"])
        sequence_output.append(seq)
        pooled_output.append(pool)

        predicted_labels = np.array(out["logits"].max(dim=1).indices.tolist())
        predicted_labels = predicted_labels[indices]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        self.visual_bert_accuracy_hist.append(accuracy)

        out = self.lxmert_forward(image, textSampleList)
        seq, pool = out["seq"], out["pooled"]
        logit.append(out["logits"])
        pooled_output.append(pool)
        sequence_output.append(seq)

        predicted_labels = np.array(out["logits"].max(dim=1).indices.tolist())
        predicted_labels = predicted_labels[indices]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        self.lxmert_accuracy_hist.append(accuracy)

        logits = torch.stack(logit).mean(0)
        logits = logits / logits.norm(dim=1, keepdim=True).clamp(min=1e-5)

        del image
        del textSampleList
        del out

        pooled_output = torch.cat(pooled_output, 1)
        # sequence_output = torch.stack(sequence_output).mean(0)
        pooled_logits = self.one_view_layer(pooled_output)
        pooled_logits = pooled_logits / pooled_logits.norm(dim=1, keepdim=True).clamp(min=1e-5)
        clean_memory()
        return logits, pooled_logits, pooled_output, sequence_output

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        labels = torch.tensor(sampleList.label, dtype=float).to(get_device())
        views = [sampleList] + [t(sampleList) for t in self.view_transforms]
        pre_logits, pooled_logits, pooled_outputs, sequence_outputs = [], [], [], []
        for view in views:
            pre_logit, pooled_logit, pooled_output, sequence_output = self.get_vectors(view)
            pre_logits.append(pre_logit)
            pooled_outputs.append(pooled_output)
            pooled_logits.append(pooled_logit)
            sequence_outputs.extend([seq[:, :self.n_tokens_out] for seq in sequence_output])
        del sampleList

        pooled_outputs = torch.cat(pooled_outputs, 1)
        # sequence_outputs = torch.stack(sequence_outputs).mean(0)
        pre_logits = torch.stack(pre_logits).mean(0)
        pre_logits = pre_logits / pre_logits.norm(dim=1, keepdim=True).clamp(min=1e-5)

        logits = self.final_layer(pooled_outputs)
        logits = logits / logits.norm(dim=1, keepdim=True).clamp(min=1e-5)
        view_loss = 0.0
        for pl in pooled_logits:
            view_loss += (((logits - pl)**2).mean())
        if len(pooled_logits) > 1:
            for pl1 in pooled_logits:
                for pl2 in pooled_logits:
                    view_loss += (((pl1 - pl2) ** 2).mean())
        pooled_logits = torch.stack(pooled_logits).mean(0)
        pooled_logits = pooled_logits / pooled_logits.norm(dim=1, keepdim=True).clamp(min=1e-5)

        view_loss += (((logits - pre_logits)**2).mean())
        view_loss += (((logits - pooled_logits) ** 2).mean())
        view_loss = self.view_loss_weight * view_loss
        self.view_loss_hist.append(view_loss.detach().cpu().item())

        actual_labels = np.array(labels.tolist())
        indices = actual_labels != self.label_not_present
        actual_labels = actual_labels[indices]

        predicted_labels = np.array(pre_logits.max(dim=1).indices.tolist())
        predicted_labels = predicted_labels[indices]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        self.pre_logit_accuracy_hist.append(accuracy)

        predicted_labels = np.array(logits.max(dim=1).indices.tolist())
        predicted_labels = predicted_labels[indices]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        self.logit_accuracy_hist.append(accuracy)

        predicted_labels = np.array(pooled_logits.max(dim=1).indices.tolist())
        predicted_labels = predicted_labels[indices]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        self.pooled_logit_accuracy_hist.append(accuracy)

        logits = (0.7 * logits + 0.2 * pooled_logits + 0.1 * pre_logits)
        logits, full_loss = loss_calculator(logits, labels if self.training else None, self.task, self.loss)
        self.full_loss_hist.append(full_loss.detach().cpu().item())

        predicted_labels = np.array(logits.max(dim=1).indices.tolist())
        predicted_labels = predicted_labels[indices]
        accuracy = accuracy_score(actual_labels, predicted_labels)
        self.full_accuracy_hist.append(accuracy)

        loss = full_loss + view_loss
        if self.training:
            loss += self.auc_dice_loss(logits, labels)
        return logits, pooled_outputs, sequence_outputs, loss


positive = ["positive", "great", "nice",
            # "practical", "useful", "awesome", "accurate", "impartial",
            # "optimistic", "effective", "hopeful",  "fortunate",
            # "helpful", "beneficial", "enjoyable", "welcome", "kind",
            # "pleasant", "agreeable", "delightful", "acceptable",
            "good", "neutral"]

negative = ["bad", "nasty", "terrible", "offensive",
            #"grim", "disagreeable", "unpleasant", "unwelcome",
            #"unacceptable", "upsetting", "unkind", "inferior",  "hateful", "hurtful",
            #"cruel", "hostility", "dislike", "resentment", "adverse", "awful",
            #"vile", "vicious", "repulsive", "disgusting", "prejudice",
            "hate", "bias", "hostile"]


class MLMSimCLR(MLMPretraining):
    def __init__(self, model: VilBertVisualBertModelV2, dropout,
                 label_to_word: dict,
                 augment_1: Callable, augment_2: Callable,
                 temperature=0.1, low_memory=False):
        super(MLMSimCLR, self).__init__(model, None, 768, "relu", 0, True)
        hidden_size = 768
        self.aug_1 = augment_1
        self.aug_2 = augment_2
        self.model = model
        self.aug_time = []
        self.model_time = []
        self.low_memory = low_memory
        self.temperature = temperature
        self.label_to_word = label_to_word
        self.label_not_present = -1  # If label not present then we will mask.

        in_dims = 768
        n_tokens_in = model.max_seq_length
        tokenizer = model.text_processor._tokenizer
        self.mask_token = self.model.text_processor._tokenizer.mask_token
        self.tokenizer = tokenizer
        self.n_tokens_in = n_tokens_in
        self.mlms = nn.ModuleList()
        self.num_seqs = 4 * ((len(model.view_transforms) + 1)) + 1
        for i in range(self.num_seqs):
            mlm = BertLMPredictionHead(hidden_size, tokenizer.vocab_size, "relu", n_tokens_in, low_memory=True)
            if i == self.num_seqs - 1:
                mlm = BertLMPredictionHead(hidden_size, tokenizer.vocab_size, "relu", n_tokens_in, low_memory=False)
            self.mlms.append(mlm)

        self.simclr_layer = nn.ModuleList()
        self.num_pools = self.num_seqs + 1
        for i in range(self.num_pools):
            if i == self.num_pools - 1:
                lin0 = nn.Linear(model.pooled_dims, hidden_size)
            else:
                lin0 = nn.Linear(in_dims, hidden_size)
            init_fc(lin0, "leaky_relu")
            lin = nn.Linear(hidden_size, hidden_size)
            init_fc(lin, "linear")
            self.simclr_layer.append(nn.Sequential(nn.Dropout(dropout), lin0, nn.LeakyReLU(), lin))
        self.loss = nn.CrossEntropyLoss()
        self.mlm_accuracy_hist = defaultdict(list)
        self.simclr_accuracy_hist = defaultdict(list)
        self.mlm_loss_hist = defaultdict(list)
        self.simclr_loss_hist = defaultdict(list)
        self.target_accuracy_hist = defaultdict(list)
        self.target_loss_hist = defaultdict(list)

    def simclr_one_sequence(self, pool1, pool2, midx):
        x1 = self.simclr_layer[midx](pool1)
        x2 = self.simclr_layer[midx](pool2)

        x1 = x1 / x1.norm(dim=1, keepdim=True).clamp(min=1e-5)
        x2 = x2 / x2.norm(dim=1, keepdim=True).clamp(min=1e-5)

        xsiz = x1.size(0)
        x1 = torch.cat((x1, x2), 0)
        x2 = x1.transpose(0, 1)
        if self.low_memory:
            x = checkpoint(torch.matmul, x1, x2)
        else:
            x = x1.mm(x2)  # batch x batch
        x = x - (torch.eye(x.size(0), device=x.device) * 1000)
        labels = torch.cat((torch.arange(xsiz, 2 * xsiz, device=x.device, dtype=torch.long), torch.arange(0, xsiz, device=x.device, dtype=torch.long)))
        x = x / self.temperature
        loss = self.loss(x, labels)
        x = torch.softmax(x.detach(), 1)
        predictions = x.max(dim=1).indices
        accuracy = accuracy_score(labels.cpu(), predictions.cpu())
        self.simclr_accuracy_hist[midx].append(accuracy)
        self.simclr_loss_hist[midx].append(float(loss.cpu().detach()))
        return loss

    def mlm_one_sequence(self, seq1, seq2,
                             input_ids_1, attention_mask_1,
                             input_ids_2, attention_mask_2,
                             midx):
        mlm = self.mlms[midx]
        batch_size = input_ids_1.size(0)
        loss_1, accuracy_1, input_ids_1, predictions_1 = mlm(seq1, input_ids_1, attention_mask_1)
        loss_2, accuracy_2, input_ids_2, predictions_2 = mlm(seq2, input_ids_2, attention_mask_2)
        accuracy = (accuracy_1 + accuracy_2)/2
        loss = (loss_1 + loss_2)/2
        predictions_1 = predictions_1.view(batch_size, -1)
        predictions_2 = predictions_2.view(batch_size, -1)

        predictions_1 = self.tokenizer.batch_decode(predictions_1.tolist(), skip_special_tokens=True)
        predictions_2 = self.tokenizer.batch_decode(predictions_2.tolist(), skip_special_tokens=True)
        predictions_1 = [p.split()[0] for p in predictions_1]
        predictions_2 = [p.split()[0] for p in predictions_2]

        p1s = []
        for p in predictions_1:
            found = False
            for k, v in self.label_to_word.items():
                if p in v:
                    p1s.append(k)
                    found = True
            if not found:
                p1s.append(self.label_not_present)

        p2s = []
        for p in predictions_2:
            found = False
            for k, v in self.label_to_word.items():
                if p in v:
                    p2s.append(k)
                    found = True
            if not found:
                p2s.append(self.label_not_present)

        p1s = torch.tensor(p1s, dtype=torch.long)
        p2s = torch.tensor(p2s, dtype=torch.long)

        self.mlm_accuracy_hist[midx].append(accuracy)
        self.mlm_loss_hist[midx].append(float(loss.cpu().detach()))
        return [p1s, p2s, loss]

    def add_label(self, sampleList):
        texts = []
        for text, label in zip(sampleList["text"], sampleList["label"]):
            if label != self.label_not_present:
                lt = random.sample(self.label_to_word[label], k=1)[0]
            else:
                lt = self.mask_token

            texts.append(lt + " " + str(text))
        sampleList["text"] = texts
        return sampleList

    def forward(self, x):
        x1 = self.aug_1(x.copy())
        x2 = self.aug_2(x.copy())

        x1 = self.add_label(x1)
        x2 = self.add_label(x2)

        input_ids_1, attention_mask_1 = self.tokenise(x1["text"])
        input_ids_2, attention_mask_2 = self.tokenise(x2["text"])

        x1 = self.model(x1)
        x2 = self.model(x2)

        seq1 = x1[2]
        seq2 = x2[2]
        pool1 = x1[1]
        pool2 = x2[1]
        logits1 = x1[0]
        logits2 = x2[0]
        loss = (x1[3] + x2[3]) / 2

        mlm_losses = 0.0
        simclr_losses = 0.0

        predicted_labels = []
        for midx, (seq1, seq2) in enumerate(zip(list(seq1), list(seq2))):
            p1s, p2s, mlm_loss = self.mlm_one_sequence(seq1, seq2, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, midx)
            predicted_labels.extend([p1s, p2s])
            mlm_losses += (0.2 * mlm_loss)
            smloss = self.simclr_one_sequence(seq1[:, :1].squeeze(), seq2[:, :1].squeeze(), midx)
            simclr_losses += (0.2 * smloss)

        predicted_labels = torch.stack(predicted_labels).type(torch.float).mean(0)
        seq1 = torch.stack(x1[2]).mean(0)
        seq2 = torch.stack(x2[2]).mean(0)
        p1s, p2s, mlm_loss = self.mlm_one_sequence(seq1, seq2, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, midx+1)
        mlm_losses += mlm_loss
        smloss = self.simclr_one_sequence(seq1[:, :1].squeeze(), seq2[:, :1].squeeze(), midx+1)
        simclr_losses += smloss

        smloss = self.simclr_one_sequence(pool1, pool2, midx+2)
        simclr_losses += smloss

        loss = loss + mlm_losses + simclr_losses + ((logits1 - logits2)**2).mean()
        predicted_labels = torch.stack([p1s.type(torch.float), p2s.type(torch.float), predicted_labels]).mean(0)
        predicted_labels = torch.cat((predicted_labels.unsqueeze(1), (1 - predicted_labels).unsqueeze(1)), 1)
        logits = (logits1 + logits2 + predicted_labels.to(get_device())) / 3

        return logits, (pool1 + pool2)/2, (seq1+seq2)/2, loss


class MLMOnly(MLMPretraining):
    def __init__(self, model: VilBertVisualBertModelV2, dropout,
                 label_to_word: dict,
                 augment_1: Callable, augment_2: Callable,
                 temperature=0.1, low_memory=False):
        super(MLMOnly, self).__init__(model, None, 768, "relu", 0, True)
        hidden_size = 768
        self.aug_1 = augment_1
        self.aug_2 = augment_2
        self.model = model
        self.aug_time = []
        self.model_time = []
        self.low_memory = low_memory
        self.temperature = temperature
        self.label_to_word = label_to_word
        self.label_not_present = -1  # If label not present then we will mask.

        n_tokens_in = model.max_seq_length
        tokenizer = model.text_processor._tokenizer
        self.mask_token = self.model.text_processor._tokenizer.mask_token
        self.tokenizer = tokenizer
        self.n_tokens_in = n_tokens_in
        self.mlms = nn.ModuleList()
        self.num_seqs = 4 * ((len(model.view_transforms) + 1)) + 1
        for i in range(self.num_seqs):
            mlm = BertLMPredictionHead(hidden_size, tokenizer.vocab_size, "relu", n_tokens_in, low_memory=False)
            if i == self.num_seqs - 1:
                mlm = BertLMPredictionHead(hidden_size, tokenizer.vocab_size, "relu", n_tokens_in, low_memory=False)
            self.mlms.append(mlm)

        self.loss = nn.CrossEntropyLoss()
        self.mlm_accuracy_hist = defaultdict(list)
        self.mlm_loss_hist = defaultdict(list)
        self.target_accuracy_hist = defaultdict(list)
        self.target_loss_hist = defaultdict(list)

    def mlm_one_sequence(self, seq1, seq2,
                             input_ids_1, attention_mask_1,
                             input_ids_2, attention_mask_2,
                             midx):
        mlm = self.mlms[midx]
        batch_size = input_ids_1.size(0)
        loss_1, accuracy_1, input_ids_1, predictions_1 = mlm(seq1, input_ids_1, attention_mask_1)
        loss_2, accuracy_2, input_ids_2, predictions_2 = mlm(seq2, input_ids_2, attention_mask_2)
        accuracy = (accuracy_1 + accuracy_2)/2
        loss = (loss_1 + loss_2)/2
        predictions_1 = predictions_1.view(batch_size, -1)
        predictions_2 = predictions_2.view(batch_size, -1)

        predictions_1 = self.tokenizer.batch_decode(predictions_1.tolist(), skip_special_tokens=True)
        predictions_2 = self.tokenizer.batch_decode(predictions_2.tolist(), skip_special_tokens=True)
        predictions_1 = [p.split()[0] for p in predictions_1]
        predictions_2 = [p.split()[0] for p in predictions_2]

        p1s = []
        for p in predictions_1:
            found = False
            for k, v in self.label_to_word.items():
                if p in v:
                    p1s.append(k)
                    found = True
            if not found:
                p1s.append(self.label_not_present)

        p2s = []
        for p in predictions_2:
            found = False
            for k, v in self.label_to_word.items():
                if p in v:
                    p2s.append(k)
                    found = True
            if not found:
                p2s.append(self.label_not_present)

        p1s = torch.tensor(p1s, dtype=torch.long)
        p2s = torch.tensor(p2s, dtype=torch.long)

        self.mlm_accuracy_hist[midx].append(accuracy)
        self.mlm_loss_hist[midx].append(float(loss.cpu().detach()))
        return [p1s, p2s, loss]

    def add_label(self, sampleList):
        texts = []
        for text, label in zip(sampleList["text"], sampleList["label"]):
            if label != self.label_not_present:
                lt = random.sample(self.label_to_word[label], k=1)[0]
            else:
                lt = self.mask_token

            texts.append(lt + " " + str(text))
        sampleList["text"] = texts
        return sampleList

    def forward(self, x):
        x1 = self.aug_1(x.copy())
        x2 = self.aug_2(x.copy())

        x1 = self.add_label(x1)
        x2 = self.add_label(x2)

        input_ids_1, attention_mask_1 = self.tokenise(x1["text"])
        input_ids_2, attention_mask_2 = self.tokenise(x2["text"])

        x1 = self.model(x1)
        x2 = self.model(x2)

        seq1 = x1[2]
        seq2 = x2[2]
        pool1 = x1[1]
        pool2 = x2[1]
        logits1 = x1[0]
        logits2 = x2[0]
        loss = (x1[3] + x2[3]) / 2

        mlm_losses = 0.0

        predicted_labels = []
        for midx, (seq1, seq2) in enumerate(zip(list(seq1), list(seq2))):
            p1s, p2s, mlm_loss = self.mlm_one_sequence(seq1, seq2, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, midx)
            predicted_labels.extend([p1s, p2s])
            mlm_losses += (0.2 * mlm_loss)

        predicted_labels = torch.stack(predicted_labels).type(torch.float).mean(0)
        seq1 = torch.stack(x1[2]).mean(0)
        seq2 = torch.stack(x2[2]).mean(0)
        p1s, p2s, mlm_loss = self.mlm_one_sequence(seq1, seq2, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, midx+1)
        mlm_losses += mlm_loss

        loss = loss + mlm_losses + ((logits1 - logits2)**2).mean()
        predicted_labels = torch.stack([p1s.type(torch.float), p2s.type(torch.float), predicted_labels]).mean(0)
        predicted_labels = torch.cat((predicted_labels.unsqueeze(1), (1 - predicted_labels).unsqueeze(1)), 1)
        logits = (logits1 + logits2 + predicted_labels.to(get_device())) / 3

        return logits, (pool1 + pool2)/2, (seq1+seq2)/2, loss


class MLMOnlyV2(MLMPretraining):
    def __init__(self, model: VilBertVisualBertModelV2, dropout,
                 label_to_word: dict,
                 augment_1: Callable,
                 mlm_loss_weight=0.01,
                 add_caption=False,
                 add_detected_objects=False,
                 add_before=True,
                 caption_object_file=None,
                 low_memory=False):
        super(MLMOnlyV2, self).__init__(model, None, 768, "relu", 0, True)
        self.add_caption = add_caption
        self.add_detected_objects = add_detected_objects
        self.add_before = add_before
        if caption_object_file is not None and (add_caption or add_detected_objects):
            assert os.path.exists(caption_object_file)
            import pandas as pd
            df = pd.read_csv(caption_object_file)
            id2_caption_objects = {row["id"]: dict(row) for r_id, row in df.iterrows()}
            self.id2_caption_objects = id2_caption_objects

        hidden_size = 768
        mlm_hidden_size = 128
        self.aug_1 = augment_1
        self.model = model
        self.aug_time = []
        self.model_time = []
        self.low_memory = low_memory
        self.mlm_loss_weight = mlm_loss_weight
        self.label_to_word = label_to_word
        self.label_not_present = -1  # If label not present then we will mask.

        n_tokens_in = model.max_seq_length
        self.n_tokens_in = n_tokens_in
        tokenizer = model.text_processor._tokenizer
        self.mask_token = self.model.text_processor._tokenizer.mask_token
        self.sep_token = self.model.text_processor._tokenizer.sep_token
        self.tokenizer = tokenizer
        self.mlm = BertLMPredictionHead(mlm_hidden_size, tokenizer.vocab_size, "relu", n_tokens_in, low_memory=False)

        self.mlm_transforms = nn.ModuleList()
        for i in range(5):
            fc = nn.Linear(hidden_size, mlm_hidden_size)
            init_fc(fc, "linear")
            self.mlm_transforms.append(fc)

        self.loss = nn.CrossEntropyLoss()
        self.mlm_accuracy_hist = defaultdict(list)
        self.mlm_loss_hist = defaultdict(list)

        self.mlm_overall_loss_hist = list()
        self.model_loss_hist = list()
        self.overall_loss_hist = list()

        self.mlm_overall_accuracy_hist = list()
        self.target_accuracy_hist = list()
        self.model_accuracy_hist = list()

    def mlm_one_sequence(self, seq1, input_ids_1, attention_mask_1, midx):
        batch_size = input_ids_1.size(0)
        loss, accuracy, input_ids_1, predictions_1 = self.mlm(seq1, input_ids_1, attention_mask_1)
        predictions_1 = predictions_1.view(batch_size, -1)
        predictions_1 = self.tokenizer.batch_decode(predictions_1.tolist(), skip_special_tokens=True)
        predictions_1 = [p.split() for p in predictions_1]
        predictions_1 = [p[0] if len(p) != 0 else self.mask_token for p in predictions_1]

        p1s = []
        for p in predictions_1:
            found = False
            for k, v in self.label_to_word.items():
                if p in v:
                    p1s.append(k)
                    found = True
            if not found:
                p1s.append(self.label_not_present)

        p1s = torch.tensor(p1s, dtype=torch.long)

        self.mlm_accuracy_hist[midx].append(accuracy)
        self.mlm_loss_hist[midx].append(float(loss))
        return [p1s, loss]

    def add_label(self, sampleList):
        texts = []
        for text, label in zip(sampleList["text"], sampleList["label"]):
            if label != self.label_not_present:
                lt = random.sample(self.label_to_word[label], k=1)[0]
            else:
                lt = self.mask_token

            texts.append(lt + " " + str(text))
        sampleList["text"] = texts
        return sampleList

    def add_objects_caption(self, sampleList):
        id2text = list(zip(sampleList["id"], sampleList["text"]))
        final_texts = sampleList["text"]
        if self.add_caption:
            final_texts = [text + f" {self.sep_token} " + self.id2_caption_objects[t_id]["caption"] for t_id, text in id2text]
        if self.add_detected_objects:
            final_texts = [text + f" {self.sep_token} " + self.id2_caption_objects[t_id]["objects"] for t_id, text in id2text]
        sampleList["text"] = final_texts
        return sampleList

    def forward(self, x):
        x1 = x
        x1 = self.add_label(x1)
        if self.add_before:
            x1 = self.add_objects_caption(x1)

        # "[SEP]"
        x = x1
        x1 = self.model(x1)
        if not self.add_before:
            x = self.add_objects_caption(x)
        input_ids_1, attention_mask_1 = self.tokenise(x["text"])

        seq1 = x1[2]
        pool1 = x1[1]
        logits1 = x1[0]
        loss = x1[3]

        mlm_losses = 0.0

        predicted_labels = []
        bad_mlm_indices = None
        for midx, sequence in enumerate(list(seq1)):
            transform_idx = midx % 4
            sequence = self.mlm_transforms[transform_idx](sequence)
            p1s, mlm_loss = self.mlm_one_sequence(sequence, input_ids_1, attention_mask_1, transform_idx)
            bad_mlm_index = p1s == self.label_not_present
            if bad_mlm_indices is not None:
                bad_mlm_indices = torch.logical_or(bad_mlm_indices, bad_mlm_index)
            else:
                bad_mlm_indices = bad_mlm_index
            predicted_labels.extend([p1s])
            mlm_losses += (0.25 * mlm_loss)

        seq1 = self.mlm_transforms[-1](torch.stack(x1[2]).mean(0))
        p1s, mlm_loss = self.mlm_one_sequence(seq1, input_ids_1, attention_mask_1, 4)
        bad_mlm_indices = torch.logical_or(bad_mlm_indices, p1s == self.label_not_present)
        mlm_losses += mlm_loss
        mlm_losses = self.mlm_loss_weight * mlm_losses

        self.mlm_overall_loss_hist.append(float(mlm_losses))
        self.model_loss_hist.append(float(loss))
        loss = loss + mlm_losses
        self.overall_loss_hist.append(float(loss))

        predicted_labels = torch.stack([p1s.type(torch.float)]+[pl.type(torch.float) for pl in predicted_labels]).mean(0)
        predicted_labels = torch.cat((predicted_labels.unsqueeze(1), (1 - predicted_labels).unsqueeze(1)), 1)
        predicted_labels = predicted_labels.to(get_device())
        predicted_indices = torch.logical_not(bad_mlm_indices).to(get_device())
        logits = logits1
        logits[predicted_indices] = (logits1[predicted_indices] + predicted_labels[predicted_indices]) / 2

        actual_labels = np.array(x["label"])
        predicted_labels = predicted_labels.max(dim=1).indices.cpu().numpy()
        indices = actual_labels != self.label_not_present
        accuracy = accuracy_score(actual_labels[indices], predicted_labels[indices])
        self.mlm_overall_accuracy_hist.append(accuracy)

        predicted_labels = np.array(logits1.max(dim=1).indices.tolist())
        accuracy = accuracy_score(actual_labels[indices], predicted_labels[indices])
        self.model_accuracy_hist.append(accuracy)

        predicted_labels = np.array(logits.max(dim=1).indices.tolist())
        accuracy = accuracy_score(actual_labels[indices], predicted_labels[indices])
        self.target_accuracy_hist.append(accuracy)
        return logits, pool1, seq1, loss


def make_plots(model: VilBertVisualBertModelV2, mlm_model: MLMOnlyV2, logy=False, exclude_from_start=0, smoothing=0):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def make_plot(title, data, x, y, hue=None):
        if smoothing > 1:
            def agg_fn(df):
                df[y] = df[y].rolling(smoothing, center=False, min_periods=1).mean()
                return df

            data = data.groupby([hue], group_keys=False).apply(agg_fn)
        plt.figure(figsize=(12, 10))
        sns.lineplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        if logy:
            plt.semilogy()
        if exclude_from_start > 0:
            plt.xlim(exclude_from_start, plt.xlim()[1])
        plt.show()

    model_losses = model.full_loss_hist + model.view_loss_hist
    x = list(range(len(model.full_loss_hist)))
    model_losses_x = x + x
    model_losses_hues = (["full"] * len(x)) + (["view"] * len(x))
    losses = pd.DataFrame({"batch": model_losses_x, "loss": model_losses, "loss_source": model_losses_hues})
    make_plot("Full Loss vs View Loss", data=losses, x="batch", y="loss", hue="loss_source")

    model_acc = model.pre_logit_accuracy_hist + model.pooled_logit_accuracy_hist + model.logit_accuracy_hist + model.full_accuracy_hist
    x = list(range(len(model.pre_logit_accuracy_hist)))
    model_acc_x = x + x + x + x
    model_acc_hues = (["pre_logit"] * len(x)) + (["pooled_logit"] * len(x)) + (["logit"] * len(x)) + (["full"] * len(x))
    acc = pd.DataFrame({"batch": model_acc_x, "accuracy": model_acc, "logit_source": model_acc_hues})
    make_plot("Different Logits Accuracy", data=acc, x="batch", y="accuracy", hue="logit_source")

    model_acc = model.vilbert_accuracy_hist + model.mmbt_region_accuracy_hist + model.visual_bert_accuracy_hist + model.lxmert_accuracy_hist
    x = list(range(len(model.vilbert_accuracy_hist))) + list(range(len(model.mmbt_region_accuracy_hist))) + list(range(len(model.visual_bert_accuracy_hist))) + list(range(len(model.lxmert_accuracy_hist)))
    model_acc_x = x
    model_acc_hues = (["vilbert"] * len(model.vilbert_accuracy_hist)) + (["mmbt_region"] * len(model.mmbt_region_accuracy_hist)) + (["visual_bert"] * len(model.visual_bert_accuracy_hist)) + (["lxmert"] * len(model.lxmert_accuracy_hist))
    acc = pd.DataFrame({"batch": model_acc_x, "accuracy": model_acc, "model": model_acc_hues})
    make_plot("Different Models Accuracy", data=acc, x="batch", y="accuracy", hue="model")

    # MLM model
    if mlm_model is None:
        return

    # MLM model: Per model MLM accuracy and loss
    # MLM model: Overall mlm, model, final loss
    # MLM model: Overall mlm acc, model acc, final acc

    model_losses = mlm_model.mlm_loss_hist[0] + mlm_model.mlm_loss_hist[1] + mlm_model.mlm_loss_hist[2] + mlm_model.mlm_loss_hist[3]
    x = list(range(len(mlm_model.mlm_loss_hist[0])))
    model_losses_x = x + x + x + x
    model_losses_hues = (["vilbert"] * len(x)) + (["mmbt_region"] * len(x)) + (["visual_bert"] * len(x)) + (["lxmert"] * len(x))
    losses = pd.DataFrame({"batch": model_losses_x, "mlm_loss": model_losses, "mlm_source": model_losses_hues})
    make_plot("MLM Loss by model", data=losses, x="batch", y="mlm_loss", hue="mlm_source")

    model_losses = mlm_model.mlm_accuracy_hist[0] + mlm_model.mlm_accuracy_hist[1] + mlm_model.mlm_accuracy_hist[2] + mlm_model.mlm_accuracy_hist[3]
    x = list(range(len(mlm_model.mlm_accuracy_hist[0])))
    model_losses_x = x + x + x + x
    model_losses_hues = (["vilbert"] * len(x)) + (["mmbt_region"] * len(x)) + (["visual_bert"] * len(x)) + (["lxmert"] * len(x))
    losses = pd.DataFrame({"batch": model_losses_x, "mlm_accuracy": model_losses, "mlm_source": model_losses_hues})
    make_plot("MLM Accuracy by model", data=losses, x="batch", y="mlm_accuracy", hue="mlm_source")

    # MLM Combined Loss and Acc
    model_losses = mlm_model.mlm_loss_hist[4] + mlm_model.mlm_accuracy_hist[4]
    x = list(range(len(mlm_model.mlm_loss_hist[4]))) + list(range(len(mlm_model.mlm_accuracy_hist[4])))
    model_losses_hues = (["Loss"] * len(mlm_model.mlm_loss_hist[4])) + (["Accuracy"] * len(mlm_model.mlm_accuracy_hist[4]))
    losses = pd.DataFrame({"batch": x, "MLM": model_losses, "Loss/Accuracy": model_losses_hues})
    make_plot("MLM Loss/Accuracy Overall", data=losses, x="batch", y="MLM", hue="Loss/Accuracy")


    model_losses = mlm_model.mlm_overall_loss_hist + mlm_model.model_loss_hist + mlm_model.overall_loss_hist
    x = list(range(len(mlm_model.overall_loss_hist)))
    model_losses_x = x + x + x
    model_losses_hues = (["MLM"] * len(x)) + (["Model"] * len(x)) + (["Total"] * len(x))
    losses = pd.DataFrame({"batch": model_losses_x, "Losses": model_losses, "Source": model_losses_hues})
    make_plot("Overall Losses", data=losses, x="batch", y="Losses", hue="Source")

    model_losses = mlm_model.mlm_overall_accuracy_hist + mlm_model.model_accuracy_hist + mlm_model.target_accuracy_hist
    x = list(range(len(mlm_model.overall_loss_hist)))
    model_losses_x = x + x + x
    model_losses_hues = (["MLM"] * len(x)) + (["Model"] * len(x)) + (["Total"] * len(x))
    losses = pd.DataFrame({"batch": model_losses_x, "Accuracy": model_losses, "Source": model_losses_hues})
    make_plot("Overall Accuracy", data=losses, x="batch", y="Accuracy", hue="Source")









