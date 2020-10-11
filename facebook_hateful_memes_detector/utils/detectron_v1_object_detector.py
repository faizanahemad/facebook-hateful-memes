import gc
import os
import sys
from collections import defaultdict, Counter
from random import random
from time import sleep
from typing import List, Callable

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from .globals import get_device, build_cache, set_global, get_global

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{DIR}/vqa-maskrcnn-benchmark')


def persistent_caching_fn(fn, name, check_cache_exists=False, cache_dir=None, cache_allow_writes=True, retries=5) -> Callable:
    wait_time = 0.25
    random_time = 0.25
    cache_dir = get_global("cache_dir") if cache_dir is None else cache_dir
    try:
        cache_allow_writes = get_global("cache_allow_writes")
    except:
        pass

    try:
        cache_stats = get_global("cache_stats")
    except:
        cache_stats = defaultdict(Counter)
        set_global("cache_stats", cache_stats)

    from diskcache import Cache
    import joblib
    if check_cache_exists:
        assert os.path.exists(cache_dir) and os.path.isdir(cache_dir)
        cache_file = os.path.join(cache_dir, "cache.db")
        assert os.path.exists(cache_file) and os.path.isfile(cache_file)
    else:
        if os.path.exists(cache_dir):
            assert os.path.isdir(cache_dir)
        else:
            os.mkdir(cache_dir)
    args = dict(eviction_policy='none', sqlite_cache_size=2 ** 16, sqlite_mmap_size=2 ** 28, disk_min_file_size=2 ** 18)
    cache = Cache(cache_dir, **args)
    try:
        import inspect
        fnh = joblib.hashing.hash(name, 'sha1') + joblib.hashing.hash(inspect.getsourcelines(fn)[0], 'sha1') + joblib.hashing.hash(fn.__name__, 'sha1')
    except Exception as e:
        try:
            fnh = joblib.hashing.hash(name, 'sha1') + joblib.hashing.hash(fn.__name__, 'sha1')
        except Exception as e:
            fnh = joblib.hashing.hash(name, 'sha1')

    def build_hash(*args, **kwargs):
        hsh = fnh + joblib.hashing.hash(args, 'sha1')
        if len(kwargs) > 0:
            hsh = hsh + joblib.hashing.hash(kwargs, 'sha1')
        return hsh

    def read_hash(hsh):
        for retry in range(retries):
            try:
                r = cache[hsh]
                cache_stats[name]["hit"] += 1
                return r
            except KeyError as ke:
                cache_stats[name]["key_error"] += 1
                sleep(wait_time + random() * random_time)
                return "ke"
            except Exception as e:
                sleep(wait_time * (retry + 1) + random() * random_time)
                cache_stats[name]["read_exception"] += 1
                cache_stats[name]["read_retries"] += 1
        cache_stats[name]["read-return-none"] += 1
        return None


    def cfn(*args, **kwargs):
        ignore_cache = kwargs.pop("ignore_cache", False)
        if ignore_cache:
            r = fn(*args, **kwargs)
            return r
        hsh = build_hash(*args, **kwargs)

        cache_stats[name]["called"] += 1

        r = read_hash(hsh)
        if r is not None and r != "ke":
            return r

        if r is None:
            r = fn(*args, **kwargs)
            cache_stats[name]["re-compute"] += 1
            cache_stats[name]["re-compute-cache-busy-no-write"] += 1
            return r

        r = fn(*args, **kwargs)  # r is not None and there was key-error so we need to calculate the key and put in cache
        cache_stats[name]["re-compute"] += 1
        if cache_allow_writes:
            for retry in range(retries * 2):
                try:
                    sleep(wait_time + random() * random_time)
                    cache[hsh] = r
                    cache_stats[name]["writes"] += 1
                    return r
                except:
                    cache_stats[name]["write_exception"] += 1
                    sleep(wait_time * (retry + 1) + random() * random_time)
                    cache_stats[name]["write_retries"] += 1
        return r

    return cfn


class LXMERTFeatureExtractor:
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), do_autocast=False):
        """
        Refer https://github.com/airsplay/py-bottom-up-attention
        :param device:
        :param do_autocast:
        """
        from detectron2.config import get_cfg
        cfg = get_cfg()
        cfg.merge_from_file(f"{DIR}/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
        cfg.MODEL.DEVICE = str(device)
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        # VG Weight
        cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
        self.cfg = cfg
        self.do_autocast = do_autocast
        self.device = device

    def __call__(self, url):
        if not hasattr(self.__class__, 'predictor'):
            from detectron2.engine import DefaultPredictor
            predictor = DefaultPredictor(self.cfg)
            setattr(self.__class__, "predictor", predictor)
            print(self.__class__.__name__, ": Loaded Model...")

        autocast_supported = False
        try:
            from torch.cuda.amp import autocast
            autocast_supported = "cuda" in str(self.device)
        except:
            pass
        if autocast_supported:
            with autocast(enabled=self.do_autocast and get_global("use_autocast")):
                detectron_features = self.doit(url, self.do_autocast and get_global("use_autocast"))
        else:
            detectron_features = self.doit(url, False)

        return detectron_features

    def get_cv2_image(self, image_path):
        if type(image_path) == np.ndarray:
            return image_path
        elif "PIL" in str(type(image_path)):
            return np.array(image_path.convert('RGB'))[:, :, ::-1]
        elif image_path.startswith('http'):
            path = requests.get(image_path, stream=True).raw
        else:
            path = image_path

        return np.array(Image.open(path).convert('RGB'))[:, :, ::-1]

    def doit(self, raw_image, autocasting=False):
        raw_image = self.get_cv2_image(raw_image)
        from detectron2.modeling.postprocessing import detector_postprocess
        from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, fast_rcnn_inference_single_image
        predictor = self.predictor
        with torch.no_grad():
            NUM_OBJECTS = 36
            raw_height, raw_width = raw_image.shape[:2]
            image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = predictor.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in predictor.model.roi_heads.in_features]
            if autocasting:
                # Half precision casting for fp16
                for box in proposal_boxes:
                    box.tensor = box.tensor.type(torch.cuda.HalfTensor)
                features = [f.type(torch.cuda.HalfTensor) for f in features]
                # Run RoI head for each proposal (RoI Pooling + Res5)

            box_features = predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)

            outputs = FastRCNNOutputs(
                predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                predictor.model.roi_heads.smooth_l1_beta,
            )
            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]

            # probs = predictor.model.roi_heads.box_predictor.predict_probs((pred_class_logits, pred_proposal_deltas,), proposals)[0]
            # boxes = predictor.model.roi_heads.box_predictor.predict_boxes((pred_class_logits, pred_proposal_deltas,), proposals)[0]

            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)

            # Note: BUTD uses raw RoI predictions,
            #       we use the predicted boxes instead.
            # boxes = proposal_boxes[0].tensor

            # NMS
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:],
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
                )
                if len(ids) == NUM_OBJECTS:
                    break

            instances = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()
            max_attr_prob = max_attr_prob[ids].detach()
            max_attr_label = max_attr_label[ids].detach()
            instances.attr_scores = max_attr_prob
            instances.attr_classes = max_attr_label
            return instances, roi_features


class FeatureExtractor:
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]

    def __init__(self, cfg_file=f'{DIR}/detectron_model.yaml', model_file=f'{DIR}/detectron_model.pth',
                 num_features=100,
                 device=torch.device('cpu')):
        self.device = device
        self.cfg_file = cfg_file # 'model_data/detectron_model.yaml'
        self.model_file = model_file # 'model_data/detectron_model.pth'
        self.num_features = num_features
        # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, url):
        if not hasattr(self.__class__, 'detection_model'):
            detection_model = self._build_detection_model()
            setattr(self.__class__, "detection_model", detection_model)
        with torch.no_grad():
            detectron_features = self.get_detectron_features(url)

        return detectron_features

    def _build_detection_model(self):
        from maskrcnn_benchmark.utils.model_serialization import load_state_dict
        from maskrcnn_benchmark.modeling.detector import build_detection_model
        from maskrcnn_benchmark.config import cfg

        cfg.merge_from_file(self.cfg_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.model_file,
                                map_location=self.device)

        load_state_dict(model, checkpoint.pop("model"))

        model.to(self.device)
        model.eval()
        print(self.__class__.__name__, ": Loaded Model...")
        return model

    def get_pil_image(self, image_path):
        if "PIL" in str(type(image_path)):
            return image_path.convert('RGB')
        elif image_path.startswith('http'):
            path = requests.get(image_path, stream=True).raw
        else:
            path = image_path

        return Image.open(path).convert('RGB')

    def _image_transform(self, image_path):
        img = self.get_pil_image(image_path)
        im = np.array(img).astype(np.float32)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        im_info = {"width": im_width, "height": im_height}
        return img, im_scale, im_info

    def _process_feature_extraction_v2(
            self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        from maskrcnn_benchmark.layers import nms
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    # & (cls_scores[keep] > conf_thresh_tensor[keep])
                    , cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)
            cls_prob = torch.max(scores[keep_boxes][start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    # "boxes": bbox.cpu().numpy(),
                    # "num_boxes": num_boxes.item(),
                    # "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    # "image_h": im_infos[i]["height"],
                    # "image_w": im_infos[i]["width"],
                    "cls_prob": scores[keep_boxes].cpu().numpy(),
                    "max_features": num_boxes.item(),
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_path):
        from maskrcnn_benchmark.structures.image_list import to_image_list
        _ = gc.collect()
        im, im_scale, im_info = self._image_transform(image_path)
        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to(self.device)
        with torch.no_grad():
            use_autocast = False
            try:
                from torch.cuda.amp import autocast
                use_autocast = "cuda" in str(self.device)
            except:
                pass
            use_autocast = use_autocast and get_global("use_autocast")
            if use_autocast:
                with autocast(enabled=False):
                    output = self.detection_model(current_img_list)
            else:
                output = self.detection_model(current_img_list)
        feat_list, info_list = self._process_feature_extraction_v2(output, im_scales, [im_info], 'fc6')
        return feat_list[0], info_list[0]
        # return {"image_feature": feat_list[0], "image_info": info_list[0]}


class ImageCaptionFeatures:
    def __init__(self, get_img_details, device=torch.device('cpu'),
                 enable_image_captions=False, beam_size=5, sample_n=5):
        self.get_img_details = get_img_details
        self.enable_image_captions = enable_image_captions
        self.device = device
        self.beam_size = beam_size
        self.sample_n = sample_n

    def __call__(self, image):
        if not hasattr(self.__class__, "model"):
            model = self.build_model(self.enable_image_captions)
            setattr(self.__class__, "model", model)

        att_embed = self.model["att_embed"]
        encoder = self.model["encoder"]
        use_autocast = False
        try:
            from torch.cuda.amp import autocast
            use_autocast = "cuda" in str(self.device)
        except:
            pass
        use_autocast = use_autocast and get_global("use_autocast")
        with torch.no_grad():
            if use_autocast:
                with autocast(enabled=False):
                    img_feature = self.get_img_details(image)[0].to(self.device)
                    att_feats = att_embed(img_feature[None])
                    att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
                    att_masks = att_masks.unsqueeze(-2)
                    em = encoder(att_feats, att_masks)
                    return em
            else:
                img_feature = self.get_img_details(image)[0].to(self.device)
                att_feats = att_embed(img_feature[None])
                att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
                att_masks = att_masks.unsqueeze(-2)
                em = encoder(att_feats, att_masks)
                return em

    def generate_captions(self, image):
        if not hasattr(self.__class__, "model"):
            model = self.build_model(self.enable_image_captions)
            setattr(self.__class__, "model", model)

        if self.enable_image_captions:
            with torch.no_grad():
                model = self.model["model"]
                img_feature = self.get_img_details(image)[0].to(self.device)
                processed_by_model = model(img_feature.mean(0)[None], img_feature[None], mode='sample',
                                           opt={'beam_size': self.beam_size, 'sample_method': 'beam_search', 'sample_n': self.sample_n})
                sents = model.decode_sequence(processed_by_model[0])
                return sents
        else:
            raise ValueError("Error: enable_image_captions = ", self.enable_image_captions)

    def build_model(self, enable_image_captions):
        import captioning.utils.misc
        import captioning.models
        infos = captioning.utils.misc.pickle_load(open(f'{DIR}/infos_trans12-best.pkl', 'rb'))
        infos['opt'].vocab = infos['vocab']
        model = captioning.models.setup(infos['opt'])
        _ = model.to(self.device)
        _ = model.load_state_dict(torch.load(f'{DIR}/model-best.pth', map_location=self.device))
        _ = model.eval()
        att_embed = model.att_embed
        encoder = model.model.encoder
        m = dict(att_embed=att_embed, encoder=encoder)
        if enable_image_captions:
            m["model"] = model
        print(self.__class__.__name__, ": Loaded Model...")
        return m


def get_image_info_fn(enable_encoder_feats=False,
                      enable_image_captions=False,
                      device=None,
                      **kwargs):
    import gc
    import torch
    if device is not None:
        kwargs["device"] = device
    else:
        device = get_device()
        kwargs["device"] = device

    def clean_memory():
        _ = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _ = gc.collect()

    feature_extractor = FeatureExtractor(**kwargs)
    lxmert_feature_extractor = LXMERTFeatureExtractor(device)

    def get_img_details(impath):
        feats = feature_extractor(impath)
        return feats

    def get_lxmert_details(impath):
        feats = lxmert_feature_extractor(impath)
        return feats

    get_lxmert_details = persistent_caching_fn(get_lxmert_details, "get_lxmert_details")

    get_img_details = persistent_caching_fn(get_img_details, "get_img_details")

    def get_batch_img_roi_features(images):
        img_feats = [get_img_details(i)[0].squeeze() for i in images]
        clean_memory()
        return torch.stack(img_feats, 0).to(device)

    def get_batch_lxmert_roi_features(images):
        img_feats = [get_lxmert_details(i)[1].squeeze() for i in images]
        clean_memory()
        return torch.stack(img_feats, 0).to(device)


    get_encoder_feats = None
    get_image_captions = None
    get_batch_encoder_feats = None
    assert enable_encoder_feats or not enable_image_captions

    if enable_encoder_feats:
        imcm = ImageCaptionFeatures(get_img_details, device, enable_image_captions)
        get_encoder_feats = persistent_caching_fn(imcm, "get_encoder_feats")

        def get_batch_encoder_feats(images, ignore_cache: List[bool] = None):
            ignore_cache = ([False] * len(images)) if ignore_cache is None else ignore_cache
            img_feats = [get_encoder_feats(i, ignore_cache=ic).squeeze() for i, ic in zip(images, ignore_cache)]
            clean_memory()
            return torch.stack(img_feats, 0).to(device)

        if enable_image_captions:
            def get_image_captions(image_text):
                return imcm.generate_captions(image_text)

    return {"get_img_details": get_img_details, "get_encoder_feats": get_encoder_feats,
            "get_image_captions": get_image_captions,
            "feature_extractor": feature_extractor,
            "get_batch_encoder_feats": get_batch_encoder_feats, "get_lxmert_details": get_lxmert_details,
            "get_batch_img_roi_features": get_batch_img_roi_features, "get_batch_lxmert_roi_features": get_batch_lxmert_roi_features}









