import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from PIL import Image
import requests
# from ....utils import persistent_caching_fn

from types import SimpleNamespace


class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.

    https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb

    Usage:
    detr = DETRdemo(device=torch.device('cuda'), num_tokens_out=16)
    detr.show_boxes('http://images.cocodataset.org/val2017/000000039769.jpg', 8, 0.5)
    """

    def __init__(self, device, im_size=360,
                 num_tokens_out=64,
                 num_classes=91, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.num_tokens_out = num_tokens_out
        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location=device, check_hash=True)
        self.load_state_dict(state_dict)
        self.eval()
        self.to_tensor = T.Compose([
            T.Resize(im_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def read_image(cls, image_path):
        if "PIL" in str(type(image_path)):
            return image_path.convert('RGB')
        elif type(image_path) == str:
            if image_path.startswith('http'):
                image_path = requests.get(image_path, stream=True).raw
            return Image.open(image_path).convert('RGB')
        else:
            raise NotImplementedError

    def get_pil_image(self, image_path):
        if type(image_path) == torch.Tensor:
            assert len(image_path.size()) == 3
            return image_path.unsqueeze(0)
        elif "PIL" in str(type(image_path)) or type(image_path) == str:
            return self.to_tensor(self.read_image(image_path)).unsqueeze(0)
        else:
            raise NotImplementedError()


    def post_process(self, h, pred_boxes, pred_logits):
        out = torch.cat([h, pred_boxes, pred_logits], 2)
        probas = pred_logits[0, :, :-1]
        sorted_scores, sorted_indices = torch.sort(probas.max(-1).values, descending=True)
        sorted_scores, sorted_indices = sorted_scores[: self.num_tokens_out], sorted_indices[: self.num_tokens_out]

        return {'pred_logits': pred_logits[:, sorted_indices], 'h': h[:, sorted_indices], 'pred_boxes': pred_boxes[:, sorted_indices],
                "seq": out[:, sorted_indices]}


    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        inputs = self.get_pil_image(inputs)
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        pred_logits = self.linear_class(h).softmax(-1)
        pred_boxes = self.linear_bbox(h).sigmoid()
        return self.post_process(h, pred_boxes, pred_logits)



    def batch_forward(self, inputs):
        assert type(inputs) == list or type(inputs) == tuple or (type(inputs) == torch.Tensor and len(inputs.size()) == 4)

    @classmethod
    def box_cxcywh_to_xyxy(cls, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @classmethod
    def rescale_bboxes(cls, out_bbox, size):
        img_w, img_h = size
        b = cls.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def show_boxes(self, image, n_objects=8, conf=0.7):
        image = self.read_image(image)
        outputs = self.forward(image)

        pred_logits = outputs['pred_logits'][:, :n_objects]
        pred_boxes = outputs['pred_boxes'][:, :n_objects]

        probas = pred_logits[0, :, :-1]
        keep = probas.max(-1).values > conf
        boxes = self.rescale_bboxes(pred_boxes[0, keep], image.size)
        prob = probas[keep]

        # COCO classes
        CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        # colors for visualization
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()


class DETR(DETRdemo):
    def __init__(self, device, im_size=360,
                 num_tokens_out=64):
        super().__init__(device, im_size, num_tokens_out)
        self.device = device

        args = dict(dataset_file="coco", batch_size=1, backbone='resnet50', position_embedding="sine", no_aux_loss=True, device=str(device),
                    resume="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", eval=True, num_workers=1)
        args = SimpleNamespace(args)

        from .detr.models import build_model
        model, _, postprocessors = build_model(args)
        model.to(self.device)
        self.model = model

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        for p in model.parameters():
            p.requires_grad = False

    def forward(self, images):
        inputs = self.get_pil_image(images)
        out = self.model(inputs)
        print(type(out), out)
