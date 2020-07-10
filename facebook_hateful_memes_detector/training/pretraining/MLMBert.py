# Give a Text list of sentences / csv and Which bert model you want to pretrain, get a saved+pretrained bert

# csv, text column, save_file, model, seq len, batch size, lr, epochs, optimiser, optimiser params (nargs), lr_schedule,

import torch
from torch import nn
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel, AutoModelForPreTraining, AutoModelWithLMHead
import pandas as pd
import argparse


def pretrain(file: Union[pd.DataFrame, str], text_column: str, ):
    pass
