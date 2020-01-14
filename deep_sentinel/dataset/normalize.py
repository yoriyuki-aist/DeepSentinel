from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def get_mean(x: 'pd.DataFrame') -> 'pd.Series':
    """Return the mean of given DataFrame along the axis=0"""
    return x.mean(axis=0)


def get_std(x: 'pd.DataFrame') -> 'pd.Series':
    """Return the std of given DataFrame along the axis=0"""
    return x.std(axis=0)


def normalize(data: 'pd.DataFrame', mean: 'pd.Series' = None, std: 'pd.Series' = None) -> 'pd.DataFrame':
    """
    Calculate Z score with given mean and std.
    If the mean or std is None, they will be calculated from given data.
    :param data:    pd.DataFrame
    :param mean:    pd.Series
    :param std:     pd.Series
    :return: pd.DataFrame
    """
    if mean is None:
        mean = get_mean(data)
    if std is None:
        std = get_std(data)
    return data.apply(
        lambda x: (x - mean[x.name]) / std[x.name]
    )
