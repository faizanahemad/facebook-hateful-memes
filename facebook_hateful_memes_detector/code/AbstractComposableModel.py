import abc
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import torch


class AbstractComposableModel(metaclass=abc.ABCMeta):
    def compose(self, text: str, image: Union[np.ndarray, torch.Tensor]):
        pass


class AbstractComposableSeqModel(metaclass=abc.ABCMeta, AbstractComposableModel):
    pass

