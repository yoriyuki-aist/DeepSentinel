from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Tuple
    import pandas as pd


def split_category_data(df: 'pd.DataFrame') -> 'Tuple[pd.DataFrame, pd.DataFrame]':
    """
    Split given DataFrame into two DataFrame, categorical values and others.
    :param df: Whole DataFrame
    :return: Tuple of DataFrame whose values are categorical data and others
    """
    category_type = 'category'
    continuous_values = df.select_dtypes(exclude=category_type)
    discrete_values = df.select_dtypes(include=category_type)
    return continuous_values, discrete_values


def split_ndarray(arr: 'np.ndarray', train_ratio: float = 0.8) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Split given ndarray into two ndarray.
    :param arr:             numpy.ndarray to split
    :param train_ratio:     Division ratio (0 < ratio < 1)
    :return: The tuple of two ndarray.
    """
    assert 0 < train_ratio < 1, "Train ratio must be more than 0 and less than 1"
    return np.split(arr, [int(arr.shape[0]*train_ratio)])
