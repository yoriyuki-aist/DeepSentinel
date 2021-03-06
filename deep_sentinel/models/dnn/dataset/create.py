from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from chainer.datasets import DictDataset, split_dataset as _split_dataset

from .constants import (
    CURRENT_CONTINUOUS, NEXT_CONTINUOUS, CURRENT_DISCRETE, NEXT_DISCRETE
)

if TYPE_CHECKING:
    from typing import Tuple
    import pandas as pd
    from chainer.datasets import SubDataset

logger = getLogger(__name__)


def _sliding_window(x: 'np.ndarray', window: int):
    dataset_length = x.shape[0]
    if window > dataset_length:
        logger.warning("The size of `bprop_length` {} is larger than dataset size {}."
                       " Use dataset size instead.".format(window, dataset_length))
        window = dataset_length
    return [x[i:i + window] for i in range(0, dataset_length, window)]


def create_dataset(continuous: 'pd.DataFrame', discrete: 'pd.DataFrame', window: int) -> 'DictDataset':
    assert window > 0, "Window size must be greater than 0"
    continuous = continuous.values.astype(np.float32)
    values_dict = {
        CURRENT_CONTINUOUS: _sliding_window(continuous[:-1], window),
        NEXT_CONTINUOUS: _sliding_window(continuous[1:], window)
    }
    if discrete.size > 0:
        x_one_hot = discrete.values.astype(np.int32)
        values_dict[CURRENT_DISCRETE] = _sliding_window(x_one_hot[:-1], window)
        values_dict[NEXT_DISCRETE] = _sliding_window(x_one_hot[1:], window)
    return DictDataset(**values_dict)


def split_dataset(dataset: 'DictDataset', train_ratio: float = 0.8) -> 'Tuple[SubDataset, SubDataset]':
    assert 0 < train_ratio < 1, "`train_ratio` size must be greater than 0 and smaller than 1"
    split_at = int(len(dataset) * train_ratio)
    if split_at == 0:
        raise ValueError("Insufficient data set length to split. "
                         "Is the size of `bprop_length` too large or given dataset wrong?")
    return _split_dataset(dataset, split_at)
