import torchvision
import torchvision.models as models

import abc
from abc import ABC
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn as nn
import torch
import torchnlp
import torch.nn.functional as F
import fasttext
from mmf.common import SampleList, Sample
from torchnlp.word_to_vector import CharNGram
from torchnlp.word_to_vector import BPEmb

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_torchvision_classification_models, get_device, get_image_info_fn
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer, TransformerEnsembleFeaturizer
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel
from ..external.mmf import get_vilbert, get_visual_bert


class VilBertVisualBertModel(nn.Module):
    def __init__(self, model_name, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, classifier_dims,
                 final_layer_builder,
                 n_tokens_out, n_layers,
                 finetune=False,
                 **kwargs):
        super(VilBertVisualBertModel, self).__init__()
        self.model_name = model_name
        if model_name == "vilbert":
            m = get_vilbert(get_device())
        elif model_name == "visual_bert":
            m = get_visual_bert(get_device())
        else:
            raise NotImplementedError()
        self.model, self.text_processor = m["model"], m["tokenizer"]
        self.get_img_details = get_image_info_fn(enable_encoder_feats=True)["get_img_details"]
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False


        # ensemble_conf = text_ensemble_conf
        # self.featurizer = TransformerEnsembleFeaturizer(ensemble_conf, n_tokens_out, classifier_dims, internal_dims,
        #                                                 n_layers, gaussian_noise, dropout)
        #
        # self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, )
        self.finetune = finetune

    def build_sample_list(self, sampleList: SampleList):
        texts = sampleList.text
        orig_image = sampleList.original_image
        texts = [self.text_processor({"text": t}) for t in texts]
        feat_list, info_list = zip(*[self.get_img_details(im) for im in orig_image])
        samples = [Sample(dict(text=t, image_feature_0=f, image_info_0=i)) for t, f, i in zip(texts, feat_list, info_list)]
        sl = SampleList(samples)
        return sl

    def vilbert_forward(self, sample_list: SampleList):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        image_info = getattr(sample_list, "image_info_0", {})
        image_dim_variable = getattr(image_info, "max_features", None)
        image_feature_variable = getattr(sample_list, "image_feature_0", None)
        image_label_variable = getattr(sample_list, "image_labels", None)
        if image_label_variable is not None:
            image_label_variable = torch.tensor(
                image_label_variable, dtype=torch.long
            ).to(get_device())

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
            image_location, dtype=torch.float
        ).to(get_device())

        cls_prob = getattr(image_info, "cls_prob", None)
        image_target = np.array(cls_prob, dtype=np.float32)
        image_target_variable = torch.tensor(image_target, dtype=torch.float).to(get_device())

        params = {"input_ids": bert_input_ids, "attention_mask": bert_input_mask, "token_type_ids": bert_input_type_ids, "image_dim": image_dim_variable,
                  "image_feature": image_feature_variable, "image_location": image_location_variable, "image_target": image_target_variable,
                  "image_label": image_label_variable, "masked_lm_labels": getattr(sample_list, "lm_label_ids", None)}

        # Prepare Mask
        if params["image_feature"] is not None and params["image_dim"] is not None:
            image_mask = (
                torch.arange(params["image_feature"].size(-2))
                    .expand(*params["image_feature"].size()[:-1])
                    .to(get_device())
            )
            if len(params["image_dim"].size()) < len(image_mask.size()):
                params["image_dim"] = params["image_dim"].unsqueeze(-1)
                assert len(params["image_dim"].size()) == len(image_mask.size())
            image_mask = image_mask < params["image_dim"]
            params["image_attention_mask"] = image_mask.long()
        else:
            params["image_attention_mask"] = None
        params.pop("image_dim")

        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            attention_weights,
        ) = self.model.model.bert(
            params["input_ids"],
            params["image_feature"],
            params["image_location"],
            params["token_type_ids"],
            params["attention_mask"],
            params["image_attention_mask"],
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
        )

        if self.model.model.fusion_method == "sum":
            pooled_output = self.model.model.dropout(pooled_output_t + pooled_output_v)
        elif self.model.model.fusion_method == "mul":
            pooled_output = self.model.model.dropout(pooled_output_t * pooled_output_v)
        else:
            raise AssertionError

        logits = self.model.model.classifier(pooled_output).contiguous().squeeze()
        output = dict(sequence_output_t=sequence_output_t,
                      sequence_output_v=sequence_output_v,
                      pooled_output_t=pooled_output_t,
                      pooled_output_v=pooled_output_v,
                      pooled_output=pooled_output,
                      logits=logits)
        return output

    def visual_bert_forward(self, sample_list: SampleList):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids
        image_info = getattr(sample_list, "image_info_0", {})
        image_dim_variable = getattr(image_info, "max_features", None)
        image_feat_variable = getattr(sample_list, "image_feature_0", None)

        sample_list.visual_embeddings = image_feat_variable
        sample_list.image_dim = image_dim_variable
        sample_list.input_ids = bert_input_ids
        sample_list.input_mask = bert_input_mask
        sample_list.token_type_ids = bert_input_type_ids

        visual_embeddings = getattr(sample_list, "visual_embeddings", None)
        image_dim = getattr(sample_list, "image_dim", None)
        # pretraining labels
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None and image_dim is not None:
            image_mask = (
                torch.arange(visual_embeddings.size(-2))
                    .expand(*visual_embeddings.size()[:-1])
                    .to(get_device())
            )
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
                assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask < image_dim
            sample_list.image_mask = image_mask.long()
        else:
            sample_list.image_mask = None

        sample_list.position_embeddings_visual = None

        sample_list = self.model.flatten_for_bert(sample_list)

        sequence_output, pooled_output, attention_weights = self.model.model.bert(
            sample_list.input_ids,
            sample_list.attention_mask,
            sample_list.token_type_ids,
            sample_list.visual_embeddings,
            sample_list.position_embeddings_visual,
            sample_list.visual_embeddings_type,
            sample_list.image_text_alignment,
            sample_list.masked_lm_labels,
        )
        output_dict = {}
        output_dict["sequence_output"] = sequence_output
        output_dict["pooled_output"] = pooled_output
        logits = self.classifier(pooled_output).contiguous().squeeze()
        output_dict["logits"] = logits
        return output_dict

    def forward(self, sampleList: SampleList):
        texts = sampleList.text
        img = sampleList.torchvision_image
        orig_image = sampleList.original_image
        labels = sampleList.label
        sample_weights = sampleList.sample_weight

        sl = self.build_sample_list(sampleList)
        if self.model_name == "vilbert":
            out = self.vilbert_forward(sl)
        elif self.model_name == "visual_bert":
            out = self.visual_bert_forward(sl)
        else:
            raise NotImplementedError()
        print({k: v.size() for k, v in out.items()})

        vectors = dict()
        for k, m in self.tx_models.items():
            _, _, text_repr, _ = m(sampleList)
            vectors[k] = text_repr

        for k, m in self.im_models.items():
            if self.finetune_image_model:
                im_repr = m(img)
            else:
                with torch.no_grad():
                    im_repr = m(img)
            im_repr = self.im_procs[k](im_repr)
            vectors[k] = im_repr

        vectors = self.featurizer(vectors)
        logits, loss = self.final_layer(vectors, labels)
        return logits, vectors.mean(1), vectors, loss
