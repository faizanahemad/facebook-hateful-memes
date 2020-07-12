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
import random


def pretrain(file: Union[pd.DataFrame, str], text_column: str, save_file: str, model: str,
             sequence_length: int, batch_size: int, accumulation_steps: int,
             lr: float, epochs: int, optimiser, optimiser_params: Dict, lr_schedule):
    if type(file) == str:
        file = pd.read_csv(file)

    if type(file) == pd.DataFrame:
        assert text_column in file.columns


