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

from ...training import calculate_auc_dice_loss, get_auc_dice_loss
from ...utils import init_fc, GaussianNoise, stack_and_pad_tensors, get_torchvision_classification_models, get_device, get_image_info_fn, Transpose, \
    dict2sampleList, loss_calculator, get_loss_by_task, clean_memory, pad_tensor, random_word_mask, load_stored_params, LinearHead
from ..classifiers import CNN1DFeaturizer, GRUFeaturizer, TransformerFeaturizer
from ..text_models import Fasttext1DCNNModel, LangFeaturesModel
from ..external.mmf import get_vilbert, get_visual_bert, get_tokenizer, get_mmbt_region
from ..external.lxrt import get_lxrt_model
import GPUtil
import random

# TODO: From each of Vilbert/visual_bert/LXMERT/MMBT take only 32 first tokens and then 32 tokens after 96 = 64 x 4 = 256 tokens/seq
# For These 256 Seq Do a Self-attn encoder layers before decoder-ensemble head. Pretrain the self-attn decoder keeping backbones const.
# Pretrain all backbones before doing combo backbones.


def identity(x): return x


class VilBertVisualBertModel(nn.Module):
    def __init__(self, model_name: Union[List, Dict], num_classes,
                 gaussian_noise, dropout,
                 internal_dims, classifier_dims,
                 featurizer, final_layer_builder,
                 n_tokens_in,
                 n_tokens_out, n_layers,
                 loss,
                 **kwargs):
        super(VilBertVisualBertModel, self).__init__()
        self.word_masking_proba = kwargs["word_masking_proba"] if "word_masking_proba" in kwargs else 0.0
        self.attention_drop_proba = kwargs["attention_drop_proba"] if "attention_drop_proba" in kwargs else 0.0
        max_seq_length = n_tokens_in
        self.text_tokens = max_seq_length
        assert type(loss) == str and loss in ["classification", "focal", "regression", "k-classification"]
        self.task = loss
        assert max_seq_length >= 64
        self.text_processor = get_tokenizer(max_seq_length)
        n_tokens_in, pooled_dims = 0, 0
        model_name = [model_name] if type(model_name) == str else model_name
        self.model_regularizers = nn.ModuleDict()
        self.model_heads = nn.ModuleDict()
        self.bbox_swaps = kwargs.pop("bbox_swaps", 0)
        self.bbox_copies = kwargs.pop("bbox_copies", 0)
        self.bbox_gaussian_noise = GaussianNoise(kwargs.pop("bbox_gaussian_noise", 0.0))
        assert type(model_name) == dict
        for k, v in model_name.items():
            dp = nn.Dropout(v["dropout"] if "dropout" in v else 0.0)
            gn = GaussianNoise(v["gaussian_noise"] if "gaussian_noise" in v else 0.0)
            self.model_regularizers[k] = nn.Sequential(dp, gn)

        self.model_name = model_name
        if "vilbert" in model_name:
            self.vilbert = get_vilbert(get_device())
            n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + 100 + max_seq_length, 768, pooled_dims + 1024
            for p in self.vilbert.parameters():
                p.requires_grad = False
            self.model_heads["vilbert"] = LinearHead(1024, 1, num_classes, dropout, self.task)
        if "visual_bert" in model_name:
            self.visual_bert = get_visual_bert(get_device())
            n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + 100 + max_seq_length, 768, pooled_dims + 768
            for p in self.visual_bert.parameters():
                p.requires_grad = False
            self.model_heads["visual_bert"] = LinearHead(768, 1, num_classes, dropout, self.task)
        if "lxmert" in model_name:
            self.lxmert = get_lxrt_model("20", pretokenized=True, max_seq_len=max_seq_length)
            n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + max_seq_length + 36, 768, pooled_dims + 768
            self.lxmert.to(get_device())
            for p in self.lxmert.parameters():
                p.requires_grad = False
            self.model_heads["lxmert"] = LinearHead(768, 1, num_classes, dropout, self.task)

        if "mmbt_region" in model_name:
            self.mmbt_region = get_mmbt_region(get_device())
            n_tokens_in, embedding_dims, pooled_dims = n_tokens_in + 102 + max_seq_length, 768, pooled_dims + 768
            for p in self.mmbt_region.parameters():
                p.requires_grad = False
            self.model_heads["mmbt_region"] = LinearHead(768, 1, num_classes, dropout, self.task)

        if len(set(model_name.keys()) - {"vilbert", "visual_bert", "lxmert", "mmbt_region"}) > 0:
            raise NotImplementedError()

        if "vilbert" in model_name or "visual_bert" in model_name or "mmbt_region" in model_name:
            self.get_img_details = get_image_info_fn(enable_encoder_feats=False, device=get_device())["get_img_details"]

        if "lxmert" in model_name:
            self.get_lxmert_details = get_image_info_fn(enable_encoder_feats=False, device=get_device())["get_lxmert_details"]

        if featurizer == "transformer":
            n_encoders = kwargs["n_encoders"] if "n_encoders" in kwargs else n_layers
            n_decoders = kwargs["n_decoders"] if "n_decoders" in kwargs else n_layers
            self.featurizer = TransformerFeaturizer(n_tokens_in, embedding_dims, n_tokens_out,
                                                    classifier_dims,
                                                    internal_dims, n_encoders, n_decoders,
                                                    gaussian_noise, dropout, self.attention_drop_proba)
        elif featurizer == "pass":
            n_tokens_out = n_tokens_in
            print("N tokens Out = ", n_tokens_out, "Classifier Dims = ", classifier_dims, "Matches embedding_dims: ", embedding_dims == classifier_dims)
            classifier_dims = embedding_dims
        else:
            raise NotImplementedError()

        self.featurizer_type = featurizer
        if self.featurizer_type == "pass":
            self.num_classes = num_classes
            lin0 = nn.Linear(pooled_dims, pooled_dims * 2)
            init_fc(lin0, "leaky_relu")
            lin1 = nn.Linear(pooled_dims * 2, 512)
            init_fc(lin1, "leaky_relu")
            lin = nn.Linear(512, num_classes)
            init_fc(lin, "linear")
            dp = nn.Dropout(dropout)
            self.final_layer = nn.Sequential(lin0, nn.LeakyReLU(), GaussianNoise(gaussian_noise), lin1, nn.LeakyReLU(), dp, lin)
            self.loss = get_loss_by_task(loss)
        else:
            n_tokens_out = n_tokens_out + len(model_name)
            self.final_layer = final_layer_builder(classifier_dims, n_tokens_out, num_classes, dropout, **kwargs)
            if "vilbert" in model_name:
                vilbert_seq_v_conv = nn.Linear(1024, 768)
                init_fc(vilbert_seq_v_conv, "leaky_relu")
                self.vilbert_seq_v_nn = nn.Sequential(vilbert_seq_v_conv, nn.LeakyReLU())

        if "stored_model" in kwargs:
            load_stored_params(self, kwargs["stored_model"])
        self.reg_layers = [(c, c.p if hasattr(c, "p") else c.sigma) for c in self.children() if c.__class__ == GaussianNoise or c.__class__ == nn.Dropout]
        self.auc_loss_coef = kwargs.pop("auc_loss_coef", 0.0)
        self.dice_loss_coef = kwargs.pop("dice_loss_coef", 0.0)
        self.auc_method = kwargs.pop("auc_method", 1)
        self.auc_dice_loss = get_auc_dice_loss(num_classes, self.dice_loss_coef, self.auc_loss_coef, auc_method=self.auc_method)

    def get_tokens(self, texts):
        keys = ["input_ids", "input_mask", "segment_ids"]
        if self.training and self.word_masking_proba > 0:
            texts = [random_word_mask(t, self.text_processor._tokenizer, self.word_masking_proba) for t in texts]
        texts = [self.text_processor({"text": t}) for t in texts]
        texts = SampleList([Sample({k: t[k] for k in keys}) for t in texts])
        for k in keys:
            texts[k] = texts[k].to(get_device())
        return texts

    def build_lxmert_sample_list(self, orig_image, textSampleList: SampleList, mixup: List[bool]):
        imgfs = [self.get_lxmert_details(im, ignore_cache=ignore_cache) for im, ignore_cache in zip(orig_image, mixup)]
        samples = [Sample(dict(feats=pad_tensor(feats, 36),
                               boxes=pad_tensor(boxes.pred_boxes.tensor, 36),
                               masks=torch.tensor(([1] * len(feats)) + ([0] * (36 - len(feats)))).long())) for boxes, feats in imgfs]
        samples = [self.bbox_aug(s, self.bbox_swaps, self.bbox_copies, self.bbox_gaussian_noise, "lxmert") for s in samples]
        sl = SampleList(samples)
        sl.input_ids = textSampleList.input_ids
        sl.input_mask = textSampleList.input_mask
        sl.segment_ids = textSampleList.segment_ids
        return sl

    def bbox_aug(self, sample, swaps=0, copies=0, gaussian_noise=identity,
                 extractor_type="vilbert_visual_bert"):

        # TODO: Do manual inspection
        if not self.training:
            return sample
        if extractor_type == "vilbert_visual_bert":
            imf = gaussian_noise(sample["image_feature_0"])
            imi = sample["image_info_0"]
            bbox = gaussian_noise(imi["bbox"])
            cls_prob = gaussian_noise(imi["cls_prob"])
            changes = [imf, bbox, cls_prob]

            for i in range(swaps):
                swap = random.sample(range(36), 2)
                for v in changes:
                    t = v[swap[1]]
                    v[swap[1]] = v[swap[0]]
                    v[swap[0]] = t

            for i in range(copies):
                copi = random.sample(range(36), 2)
                for v in changes:
                    v[copi[0]] = v[copi[1]]

        elif extractor_type == "lxmert":
            for i in range(swaps):
                swap = random.sample(range(36), 2)
                for k, v in sample.items():
                    v = gaussian_noise(v)
                    t = v[swap[1]]
                    v[swap[1]] = v[swap[0]]
                    v[swap[0]] = t
                    sample[k] = v

            for i in range(copies):
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
        samples = [self.bbox_aug(s, self.bbox_swaps, self.bbox_copies, self.bbox_gaussian_noise, "vilbert_visual_bert") for s in samples]
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

    def vilbert_processor(self, sample_list: SampleList, labels):
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

        logits, loss = self.model_heads["vilbert"](pooled_output, labels)
        output = dict(sequence_output_t=sequence_output_t,
                      sequence_output_v=sequence_output_v,
                      pooled_output_t=pooled_output_t,
                      pooled_output_v=pooled_output_v,
                      pooled_output=pooled_output,
                      logits=logits, loss=loss)
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
        return output_dict

    def visual_bert_forward(self, sl: SampleList, labels):
        params = self.__visual_bert_preprocessing__(sl)
        del sl
        clean_memory()
        out = self.visual_bert_processor(params)
        logits, loss = self.model_heads["visual_bert"](out["pooled_output"], labels)
        out["logits"] = logits
        out["loss"] = loss
        return out

    def lxmert_forward(self, orig_image, textSampleList, labels, mixup: List[bool]):
        lx_sl = self.build_lxmert_sample_list(orig_image, textSampleList, mixup)
        for k, v in lx_sl.items():
            if type(v) == torch.Tensor:
                lx_sl[k] = v.to(get_device())
        feat_seq, pooled = self.lxmert((lx_sl.input_ids, lx_sl.input_mask, lx_sl.segment_ids,), (lx_sl.feats, lx_sl.boxes), lx_sl.masks)

        del lx_sl
        clean_memory()
        logits, loss = self.model_heads["lxmert"](pooled, labels)
        return dict(seq=torch.cat(feat_seq, 1), pooled=pooled, logits=logits, loss=loss)

    def mmbt_region_forward(self, sl: SampleList, labels):
        sl = sl.to(get_device())
        sl.image_feature_0 = sl.image_feature_0.type(torch.float)
        module_output = self.mmbt_region.model.bert(sl)
        pooled_output = module_output[1]
        output = {}
        output["sequence_output"] = module_output[0]
        output["pooled_output"] = pooled_output
        logits = None
        logits, loss = self.model_heads["mmbt_region"](pooled_output, labels)
        output["logits"] = logits
        output["loss"] = loss
        del sl
        clean_memory()
        return output

    def get_vectors(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        texts = sampleList.text
        image = sampleList.image  # orig_image = sampleList.original_image
        labels = torch.tensor(sampleList.label, dtype=float).to(get_device())
        mixup = sampleList.mixup

        textSampleList = self.get_tokens(texts)
        textSampleList.id = sampleList.id
        del sampleList
        clean_memory()
        # GPUtil.showUtilization()
        pooled_output = []
        sequence_output = []
        logits = None
        loss = 0.0
        loss_counts = 0
        logit = []
        if "vilbert" in self.model_name or "visual_bert" in self.model_name or "mmbt_region" in self.model_name:
            sl = self.build_vilbert_visual_bert_sample_list(image, textSampleList, mixup)
            if "vilbert" in self.model_name:
                out = self.vilbert_processor(sl, labels)
                if self.featurizer_type != "pass":
                    out["sequence_output_v"] = self.vilbert_seq_v_nn(out["sequence_output_v"])
                else:
                    out["sequence_output_v"] = out["sequence_output_v"][:, :, :out["sequence_output_t"].size(-1)]
                seq = torch.cat([out["sequence_output_v"], out["sequence_output_t"]], 1)
                seq = self.model_regularizers["vilbert"](seq) if "vilbert" in self.model_regularizers else seq
                sequence_output.append(seq)
                pool = out["pooled_output"]
                pool = self.model_regularizers["vilbert"](pool) if "vilbert" in self.model_regularizers else pool
                pooled_output.append(pool)
                logit.append(out["logits"])
                loss+=out["loss"]
                loss_counts+=1
                del out
                clean_memory()

            if "mmbt_region" in self.model_name:
                out = self.mmbt_region_forward(sl, labels)
                seq, pool = out["sequence_output"], out["pooled_output"]
                seq = self.model_regularizers["mmbt_region"](seq) if "mmbt_region" in self.model_regularizers else seq
                pool = self.model_regularizers["mmbt_region"](pool) if "mmbt_region" in self.model_regularizers else pool
                logit.append(out["logits"])
                loss += out["loss"]
                loss_counts += 1
                del out
                sequence_output.append(seq)
                pooled_output.append(pool)
                clean_memory()

            if "visual_bert" in self.model_name:
                out = self.visual_bert_forward(sl, labels)
                seq, pool = out["sequence_output"], out["pooled_output"]
                seq = self.model_regularizers["visual_bert"](seq) if "visual_bert" in self.model_regularizers else seq
                pool = self.model_regularizers["visual_bert"](pool) if "visual_bert" in self.model_regularizers else pool
                logit.append(out["logits"])
                loss += out["loss"]
                loss_counts += 1
                del out
                sequence_output.append(seq)
                pooled_output.append(pool)

            del sl
            clean_memory()

        if "lxmert" in self.model_name:
            out = self.lxmert_forward(image, textSampleList, labels, mixup)
            seq, pool = out["seq"], out["pooled"]
            seq = self.model_regularizers["lxmert"](seq) if "lxmert" in self.model_regularizers else seq
            pool = self.model_regularizers["lxmert"](pool) if "lxmert" in self.model_regularizers else pool
            logit.append(out["logits"])
            pooled_output.append(pool)
            sequence_output.append(seq)
            loss += out["loss"]
            loss_counts += 1
            del out
        logits = torch.stack(logit).mean(0)
        loss = loss / loss_counts

        del image
        del textSampleList

        sequenced_pooled_output = torch.cat([p.unsqueeze(1) for p in pooled_output], 1)
        pooled_output = torch.cat(pooled_output, 1) if len(pooled_output) > 1 else pooled_output[0]
        sequence_output = torch.cat(sequence_output, 1) if len(sequence_output) > 1 else sequence_output[0]
        clean_memory()
        return logits, loss, pooled_output, sequenced_pooled_output, sequence_output

    def forward(self, sampleList: SampleList):
        sampleList = dict2sampleList(sampleList, device=get_device())
        labels = torch.tensor(sampleList.label, dtype=float).to(get_device())
        sample_weights = sampleList.sample_weight
        # GPUtil.showUtilization()
        pre_logits, pre_loss, pooled_output, sequenced_pooled_output, sequence_output = self.get_vectors(sampleList)
        del sampleList
        clean_memory()

        if self.featurizer_type == "pass":
            logits = self.final_layer(pooled_output)
            logits, loss = loss_calculator(logits, labels if self.training else None, self.task, self.loss)
        else:
            vectors = self.featurizer(sequence_output)
            sequenced_pooled_output = sequenced_pooled_output[:, :, :vectors.size(-1)]
            vectors = torch.cat([vectors, sequenced_pooled_output], 1)
            logits, loss = self.final_layer(vectors, labels)
        if self.training:
            loss += self.auc_dice_loss(logits, labels)
        loss = 0.75 * loss + 0.25 * pre_loss
        logits = (0.75 * logits + 0.25 * pre_logits)
        return logits, pooled_output, sequence_output, loss
