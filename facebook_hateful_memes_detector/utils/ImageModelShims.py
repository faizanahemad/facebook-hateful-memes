from ..models.classifiers import TransformerFeaturizer
from ..utils import *
DIR = os.path.dirname(os.path.realpath(__file__))


# TODO: Use both VggFace and Resnet50-SSL 7x7, 3x3, global view -> 61
# TODO: Retrain this for AugSim, Differentiator, SimCLR objective,
# TODO: Enable Multi-view (HFlip, Zoom-in, Zoom-out, +15, -15 Rotate), Add 2 layer encoder self-attn take first 128/64 tokens
# TODO: Linear layer 2048->768 before encoder
# Don't Multi-View now.


#

class ImageModelShim(nn.Module):
    def __init__(self, resnet="resnet50_ssl", n_tokens=64, out_channels=768, n_encoders=2, dropout=0.0, gaussian_noise=0.0, attention_drop_proba=0.0, **kwargs):
        super().__init__()
        resnet_model, resnet_shape = get_torchvision_classification_models(resnet, True)
        vgg_shape = (256, 1)
        vgg_model = get_vgg_face_model()

        self.resnet_model = resnet_model
        self.vgg_model = vgg_model
        self.resnet = resnet

        gaussian_noise = GaussianNoise(gaussian_noise)

        lin = nn.Linear(resnet_shape[0], out_channels)
        init_fc(lin, "linear")
        self.resnet_reshape = nn.Sequential(nn.Dropout(dropout), lin, gaussian_noise)

        lin = nn.Linear(vgg_shape[0], out_channels)
        init_fc(lin, "linear")
        self.vgg_reshape = nn.Sequential(nn.Dropout(dropout), lin, gaussian_noise)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        half_dim = 3
        self.half_pool = nn.AdaptiveMaxPool2d(half_dim)

        self.quadrant_pool = nn.AdaptiveAvgPool2d(2)

        n_tokens_in = (resnet_shape[1] * resnet_shape[1]) + 1 + (half_dim * half_dim) + 1 + (2 * 2)
        featurizer = TransformerFeaturizer(n_tokens_in, out_channels, n_tokens,
                                           out_channels,
                                           out_channels, n_encoders, 0,
                                           gaussian_noise, dropout, attention_drop_proba)
        self.featurizer = featurizer

        if "stored_model" in kwargs and kwargs["stored_model"] is not None:
            load_stored_params(self, kwargs["stored_model"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        resnet_in = self.resnet_model(images)
        resnet_lrf = self.half_pool(resnet_in)
        resnet_quadrant = self.quadrant_pool(resnet_in)
        resnet_global = self.global_pool(resnet_in).squeeze().unsqueeze(1)
        vgg_face_in = self.vgg_reshape(self.vgg_model(images).squeeze().unsqueeze(1))

        resnet_in = resnet_in.flatten(2, 3).transpose(1, 2)  # B,C,H,W -> B,HxW,C
        resnet_lrf = resnet_lrf.flatten(2, 3).transpose(1, 2)
        resnet_quadrant = resnet_quadrant.flatten(2, 3).transpose(1, 2)

        resnet_out = self.resnet_reshape(torch.cat([resnet_global, resnet_in, resnet_lrf, resnet_quadrant], 1))

        seq = torch.cat([resnet_out, vgg_face_in], 1)
        seq = self.featurizer(seq)
        return seq


class ImageCaptioningShim(nn.Module):
    def __init__(self, n_tokens=1, n_encoders=2, dropout=0.0, gaussian_noise=0.0, attention_drop_proba=0.0, **kwargs):
        super().__init__()
        lin = nn.Linear(512, 768)
        init_fc(lin, "linear")
        self.reshape = nn.Sequential(nn.Dropout(dropout), lin, nn.LayerNorm(768))
        self.captioner = get_image_info_fn(enable_encoder_feats=True)["get_batch_encoder_feats"]
        featurizer = TransformerFeaturizer(100, 768, n_tokens,
                                           768,
                                           768, n_encoders, 0,
                                           gaussian_noise, dropout, attention_drop_proba)
        self.featurizer = featurizer

        if "stored_model" in kwargs and kwargs["stored_model"] is not None:
            load_stored_params(self, kwargs["stored_model"])

    def forward(self, images: List, ignore_cache: List[bool] = None):
        caption_features = self.captioner(images, ignore_cache)
        caption_features = self.reshape(caption_features)
        caption_features = self.featurizer(caption_features)
        return caption_features


class ImageModelShimSimple(nn.Module):
    def __init__(self, resnet="resnet18_swsl", n_tokens=64, out_channels=768, n_encoders=2, dropout=0.0, gaussian_noise=0.0, attention_drop_proba=0.0, **kwargs):
        super().__init__()
        resnet_model, resnet_shape = get_torchvision_classification_models(resnet, True)
        self.resnet_model = resnet_model
        self.resnet = resnet

        gaussian_noise = GaussianNoise(gaussian_noise)

        internal_dims = 128
        lin = nn.Linear(resnet_shape[0], internal_dims)
        init_fc(lin, "linear")
        self.resnet_reshape = nn.Sequential(nn.Dropout(dropout), lin, gaussian_noise)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        half_dim = 3
        self.half_pool = nn.AdaptiveMaxPool2d(half_dim)

        self.quadrant_pool = nn.AdaptiveAvgPool2d(2)

        featurizer = TransformerFeaturizer(n_tokens, internal_dims, n_tokens,
                                           out_channels,
                                           internal_dims, n_encoders, 0,
                                           gaussian_noise, dropout, attention_drop_proba)
        self.featurizer = featurizer
        use_autocast = False
        try:
            from torch.cuda.amp import GradScaler, autocast
            use_autocast = "cuda" in str(get_device())
        except:
            pass
        self.use_autocast = use_autocast and get_global("use_autocast")
        self.out_ln = nn.LayerNorm(out_channels, eps=1e-12)
        self.out_channels = out_channels

        if "stored_model" in kwargs and kwargs["stored_model"] is not None:
            load_stored_params(self, kwargs["stored_model"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_autocast:
            images = images.type(torch.cuda.HalfTensor)
        resnet_in = self.resnet_model(images)
        resnet_lrf = self.half_pool(resnet_in)
        resnet_quadrant = self.quadrant_pool(resnet_in)
        resnet_global = self.global_pool(resnet_in).squeeze().unsqueeze(1)

        resnet_in = resnet_in.flatten(2, 3).transpose(1, 2)  # B,C,H,W -> B,HxW,C
        resnet_lrf = resnet_lrf.flatten(2, 3).transpose(1, 2)
        resnet_quadrant = resnet_quadrant.flatten(2, 3).transpose(1, 2)

        resnet_out = self.resnet_reshape(torch.cat([resnet_global, resnet_lrf, resnet_in, resnet_quadrant, resnet_global], 1))
        seq = self.featurizer(resnet_out)
        seq = self.out_ln(seq)
        return seq




