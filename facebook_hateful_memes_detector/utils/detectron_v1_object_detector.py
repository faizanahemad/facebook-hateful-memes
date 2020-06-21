import numpy as np

import sys
import os

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from io import BytesIO
from .globals import get_device, set_device, set_cpu_as_device, set_first_gpu, memory, build_cache

DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{DIR}/vqa-maskrcnn-benchmark')


def persistent_caching_fn(fn, name, cache_dir=os.path.join(os.getcwd(), 'cache')):
    from diskcache import Cache, Index
    import joblib
    if os.path.exists(cache_dir):
        assert os.path.isdir(cache_dir)
    else:
        os.mkdir(cache_dir)
    cache = Index(cache_dir, sqlite_cache_size=2 ** 13, sqlite_mmap_size=2 ** 26)
    try:
        import inspect
        fnh = joblib.hashing.hash(name, 'sha1') + joblib.hashing.hash(inspect.getsourcelines(fn)[0], 'sha1') + joblib.hashing.hash(fn.__name__, 'sha1')
    except Exception as e:
        try:
            fnh = joblib.hashing.hash(name, 'sha1') + joblib.hashing.hash(fn.__name__, 'sha1')
        except Exception as e:
            fnh = joblib.hashing.hash(name, 'sha1')

    def cfn(*args, **kwargs):
        hsh = fnh + joblib.hashing.hash(args, 'sha1')
        if len(kwargs) > 0:
            hsh = hsh + joblib.hashing.hash(kwargs, 'sha1')
        if hsh in cache:
            try:
                return cache[hsh]
            except KeyError as ke:
                pass
        r = fn(*args, **kwargs)
        cache[hsh] = r
        return r

    return cfn



class FeatureExtractor:
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]

    def __init__(self, cfg_file=f'{DIR}/detectron_model.yaml', model_file=f'{DIR}/detectron_model.pth',
                 num_features=100,
                 device=torch.device('cpu')):
        self.device = device
        self.cfg_file = cfg_file # 'model_data/detectron_model.yaml'
        self.model_file = model_file # 'model_data/detectron_model.pth'
        self.detection_model = self._build_detection_model()
        self.num_features = num_features
        # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, url):
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
                    "boxes": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "image_h": im_infos[i]["height"],
                    "image_w": im_infos[i]["width"],
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
            output = self.detection_model(current_img_list)
        feat_list, info_list = self._process_feature_extraction_v2(output, im_scales, [im_info], 'fc6')
        return feat_list[0], info_list[0]
        # return {"image_feature": feat_list[0], "image_info": info_list[0]}


def get_image_info_fn(enable_encoder_feats=False,
                      enable_image_captions=False,
                      cachedir=None,
                      device=None,
                      **kwargs):
    import gc
    if device is not None:
        kwargs["device"] = device
    else:
        device = get_device()

    feature_extractor = FeatureExtractor(**kwargs)

    if cachedir is None:
        global memory
    else:
        memory = build_cache(cachedir)

    def get_img_details(impath):
        feats = feature_extractor(impath)
        return feats

    get_img_details = persistent_caching_fn(get_img_details, "get_img_details")


    get_encoder_feats = None
    get_image_captions = None
    get_batch_encoder_feats = None
    assert enable_encoder_feats or not enable_image_captions

    if enable_encoder_feats:
        import captioning
        import captioning.utils.misc
        import captioning.models
        infos = captioning.utils.misc.pickle_load(open(f'{DIR}/infos_trans12-best.pkl', 'rb'))
        infos['opt'].vocab = infos['vocab']
        model = captioning.models.setup(infos['opt'])
        _ = model.to(device)
        _ = model.load_state_dict(torch.load(f'{DIR}/model-best.pth', map_location=device))
        _ = model.eval()
        att_embed = model.att_embed
        encoder = model.model.encoder

        def get_encoder_feats(image_text):
            with torch.no_grad():
                img_feature = get_img_details(image_text)[0]
                att_feats = att_embed(img_feature[None])
                att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
                att_masks = att_masks.unsqueeze(-2)
                em = encoder(att_feats, att_masks)
                return em

        get_encoder_feats = persistent_caching_fn(get_encoder_feats, "get_encoder_feats")

        def get_batch_encoder_feats(images):
            img_feats = [get_encoder_feats(i).squeeze() for i in images]
            _ = gc.collect()
            return torch.stack(img_feats, 0).to(device)

        if enable_image_captions:
            def get_image_captions(image_text):
                img_feature = get_img_details(image_text)[0]
                processed_by_model = model(img_feature.mean(0)[None], img_feature[None], mode='sample',
                                           opt={'beam_size': 5, 'sample_method': 'beam_search', 'sample_n': 5})
                sents = model.decode_sequence(processed_by_model[0])
                return sents

    return {"get_img_details": get_img_details, "get_encoder_feats": get_encoder_feats,
            "get_image_captions": persistent_caching_fn(get_image_captions, "get_image_captions"),
            "feature_extractor": persistent_caching_fn(feature_extractor, "feature_extractor"),
            "get_batch_encoder_feats": get_batch_encoder_feats}









