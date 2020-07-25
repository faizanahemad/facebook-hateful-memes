
import pandas as pd
import numpy as np
import jsonlines
import seaborn as sns
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from importlib import reload
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', '{:0.3f}'.format)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings
import os
import torchvision

warnings.filterwarnings('ignore')

from facebook_hateful_memes_detector.utils.globals import set_global, get_global

set_global("cache_dir", "/home/ahemf/cache/cache" if os.path.exists("/home/ahemf/cache/cache") else os.path.join(os.getcwd(), "cache"))
set_global("dataloader_workers", 0)
set_global("use_autocast", True)
set_global("models_dir", "/home/ahemf/cache/" if os.path.exists("/home/ahemf/cache/") else os.path.join(os.getcwd(), "cache"))

from facebook_hateful_memes_detector.utils import read_json_lines_into_df, in_notebook, set_device
print(get_global("cache_dir"))
from facebook_hateful_memes_detector.models import Fasttext1DCNNModel, MultiImageMultiTextAttentionEarlyFusionModel, LangFeaturesModel, AlbertClassifer, TransformerImageModel

from facebook_hateful_memes_detector.preprocessing import TextImageDataset, my_collate, get_datasets, get_image2torchvision_transforms, TextAugment
from facebook_hateful_memes_detector.preprocessing import DefinedRotation, QuadrantCut, ImageAugment, DefinedAffine
from facebook_hateful_memes_detector.training import *
import facebook_hateful_memes_detector
reload(facebook_hateful_memes_detector)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)

from PIL import Image

img = Image.open("../data/img/03745.png")
from facebook_hateful_memes_detector.utils.detectron_v1_object_detector import FeatureExtractor

fe = FeatureExtractor()
res = fe(img)
print(res[0])
print(res[1])


