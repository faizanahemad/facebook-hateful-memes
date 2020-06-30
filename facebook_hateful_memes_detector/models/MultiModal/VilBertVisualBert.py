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

from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_torchvision_classification_models, get_device, get_image_info_fn, Transpose, \
    dict2sampleList, loss_calculator, get_loss_by_task, clean_memory, pad_tensor
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer, TransformerEnsembleFeaturizer, BasicFeaturizer, PassThroughFeaturizer
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel
from ..external.mmf import get_vilbert, get_visual_bert, get_tokenizer
from ..external.lxrt import get_lxrt_model
import GPUtil


class VilBertVisualBertModel(nn.Module):
    def __init__(self, model_name: List, num_classes,
                 gaussian_noise, dropout,
                 internal_dims, classifier_dims,
                 featurizer, final_layer_builder,
                 n_tokens_out, n_layers,
                 task,
                 finetune=False,
                 **kwargs):
        super(VilBertVisualBertModel, self).__init__()
        self.task = task
        max_seq_length = 64
        self.text_processor = get_tokenizer(max_seq_length)
        n_tokens_in, pooled_dims = 0, 0
        model_name = [model_name] if type(model_name) ==  str else model_name
        self.model_name = model_name
        if "vilbert" in model_name:
            self.vilbert = get_vilbert(get_device())
            n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + 100 + max_seq_length, 768, pooled_dims + 1024
            vilbert_seq_v_conv = nn.Conv1d(1024, 768, 1, 1, groups=8)
            init_fc(vilbert_seq_v_conv, "leaky_relu")
            self.vilbert_seq_v_nn = nn.Sequential(Transpose(), vilbert_seq_v_conv, nn.LeakyReLU(), Transpose(), nn.LayerNorm(768))
        if "visual_bert" in model_name:
            self.visual_bert = get_visual_bert(get_device())
            n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + 100 + max_seq_length, 768, pooled_dims + 768

        if "lxmert" in model_name:
            self.lxmert = get_lxrt_model("20", pretokenized=True, max_seq_len=max_seq_length)
            n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + max_seq_length + 36, 768, pooled_dims + 768
            self.lxmert.to(get_device())

        if len(model_name) > 1:
            assert featurizer == "transformer"

        if len(set(model_name) - {"vilbert", "visual_bert", "lxmert"}) > 0:
            raise NotImplementedError()

        if "vilbert" in model_name or "visual_bert" in model_name:
            self.get_img_details = get_image_info_fn(enable_encoder_feats=False, device=get_device())["get_img_details"]

        if "lxmert" in model_name:
            self.get_lxmert_details = get_image_info_fn(enable_encoder_feats=False, device=get_device())["get_lxmert_details"]

        if not finetune:
            if hasattr(self, 'model'):
                for p in self.model.parameters():
                    p.requires_grad = False
            if hasattr(self, 'vilbert'):
                for p in self.vilbert.parameters():
                    p.requires_grad = False
            if hasattr(self, 'visual_bert'):
                for p in self.visual_bert.parameters():
                    p.requires_grad = False
            if hasattr(self, 'lxmert'):
                for p in self.lxmert.parameters():
                    p.requires_grad = False

        if featurizer == "transformer":
            self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                    classifier_dims,
                                                    internal_dims, n_layers, gaussian_noise, dropout)
        elif featurizer == "pass":
            assert n_tokens_in == n_tokens_out
            assert embedding_dims == classifier_dims
            assert ("vilbert" in model_name or "visual_bert" in model_name) and "lxmert" not in model_name
        else:
            raise NotImplementedError()

        self.featurizer_type = featurizer
        if self.featurizer_type == "pass":
            self.num_classes = num_classes
            if self.num_classes != 2 or "lxmert" in model_name or len(model_name) > 1:
                lin0 = nn.Linear(pooled_dims, pooled_dims)
                init_fc(lin0, "leaky_relu")
                lin = nn.Linear(pooled_dims, num_classes)
                init_fc(lin, "linear")
                dp = nn.Dropout(dropout)
                ll = nn.LayerNorm(pooled_dims)
                self.final_layer = nn.Sequential(dp, lin0, nn.LeakyReLU(), ll, lin)
            else:
                assert finetune
            self.loss = get_loss_by_task(task)
        else:
            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, )


        # ensemble_conf = text_ensemble_conf
        # self.featurizer = TransformerEnsembleFeaturizer(ensemble_conf, n_tokens_out, classifier_dims, internal_dims,
        #                                                 n_layers, gaussian_noise, dropout)
        #

        self.finetune = finetune

    def get_tokens(self, texts):
        keys = ["input_ids", "input_mask", "segment_ids"]
        texts = [self.text_processor({"text": t}) for t in texts]
        texts = SampleList([Sample({k: t[k] for k in keys}) for t in texts])
        for k in keys:
            texts[k] = texts[k].to(get_device())
        return texts

    def build_lxmert_sample_list(self, orig_image, textSampleList: SampleList):
        imgfs = [self.get_lxmert_details(im) for im in orig_image]
        samples = [Sample(dict(feats=pad_tensor(feats, 36),
                               boxes=pad_tensor(boxes.pred_boxes.tensor, 36),
                               masks=torch.tensor(([1] * len(feats)) + ([0] * (36 - len(feats)))).long())) for boxes, feats in imgfs]
        sl = SampleList(samples)
        sl.input_ids = textSampleList.input_ids
        sl.input_mask = textSampleList.input_mask
        sl.segment_ids = textSampleList.segment_ids
        return sl


    def build_vilbert_visual_bert_sample_list(self, orig_image, textSampleList: SampleList):
        imgfs = [self.get_img_details(im) for im in orig_image]
        samples = [Sample(dict(image_feature_0=feat_list, image_info_0=info_list)) for feat_list, info_list in imgfs]
        sl = SampleList(samples)
        sl.input_ids = textSampleList.input_ids
        sl.input_mask = textSampleList.input_mask
        sl.segment_ids = textSampleList.segment_ids
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


        logits = None
        if self.featurizer_type == "pass":
            logits = self.vilbert.model.classifier(pooled_output).contiguous().squeeze()
        output = dict(sequence_output_t=sequence_output_t,
                      sequence_output_v=sequence_output_v,
                      pooled_output_t=pooled_output_t,
                      pooled_output_v=pooled_output_v,
                      pooled_output=pooled_output,
                      logits=logits)
        # GPUtil.showUtilization()
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
        clean_memory()
        logits = None
        if self.featurizer_type == "pass":
            logits = self.visual_bert.model.classifier(pooled_output).contiguous().squeeze()
        output_dict["logits"] = logits
        return output_dict


    def visual_bert_forward(self, sl: SampleList):
        params = self.__visual_bert_preprocessing__(sl)
        del sl
        clean_memory()
        out = self.visual_bert_processor(params)
        return out


    def lxmert_forward(self, orig_image, textSampleList):
        lx_sl = self.build_lxmert_sample_list(orig_image, textSampleList)
        for k, v in lx_sl.items():
            if type(v) == torch.Tensor:
                lx_sl[k] = v.to(get_device())
        feat_seq, pooled = self.lxmert((lx_sl.input_ids, lx_sl.input_mask, lx_sl.segment_ids,), (lx_sl.feats, lx_sl.boxes), lx_sl.masks)

        del lx_sl
        clean_memory()
        return torch.cat(feat_seq, 1), pooled

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        texts = sampleList.text
        image = sampleList.image # orig_image = sampleList.original_image
        labels = torch.tensor(sampleList.label, dtype=float).to(get_device())
        sample_weights = sampleList.sample_weight

        textSampleList = self.get_tokens(texts)
        del sampleList
        clean_memory()
        # GPUtil.showUtilization()
        pooled_output = []
        sequence_output = []
        logits = None
        if "vilbert" in self.model_name or "visual_bert" in self.model_name:
            sl = self.build_vilbert_visual_bert_sample_list(image, textSampleList)
            logit = []
            if "vilbert" in self.model_name:
                out = self.vilbert_processor(sl)
                if self.featurizer_type != "pass":
                    out["sequence_output_v"] = self.vilbert_seq_v_nn(out["sequence_output_v"])
                else:
                    out["sequence_output_v"] = out["sequence_output_v"][:, :, :out["sequence_output_t"].size(-1)]
                sequence_output.append(torch.cat([out["sequence_output_v"], out["sequence_output_t"]], 1))
                pooled_output.append(out["pooled_output"])
                logit.append(out["logits"])
                del out
                clean_memory()

            if "visual_bert" in self.model_name:
                out = self.visual_bert_forward(sl)
                seq, pool = out["sequence_output"], out["pooled_output"]
                logit.append(out["logits"])
                del out
                sequence_output.append(seq)
                pooled_output.append(pool)

            logits = torch.softmax(torch.stack(logit).mean(0), dim=1)


            del sl
            clean_memory()

        if "lxmert" in self.model_name:
            feat_seq, pooled = self.lxmert_forward(image, textSampleList)
            pooled_output.append(pooled)
            sequence_output.append(feat_seq)

        del image
        del textSampleList

        pooled_output = torch.cat(pooled_output, 1) if len(pooled_output) > 1 else pooled_output[0]
        sequence_output = torch.cat(sequence_output, 1) if len(sequence_output) > 1 else sequence_output[0]
        clean_memory()
        # GPUtil.showUtilization()

        if self.featurizer_type == "pass":
            if self.model.config.num_labels != self.num_classes:
                logits = self.final_layer(pooled_output)
            logits, loss = loss_calculator(logits, labels, self.task, self.loss)
        else:
            vectors = self.featurizer(sequence_output)
            logits, loss = self.final_layer(vectors, labels)
        return logits, pooled_output, sequence_output, loss
