import logging

import numpy as np
import pandas as pd
import pytest
from chainer.datasets import DictDataset

from deep_sentinel.models.dnn import dataset
from deep_sentinel.models.dnn.dataset import constants


@pytest.fixture(scope='session')
def dataset_keys():
    return [
        constants.CURRENT_CONTINUOUS,
        constants.CURRENT_DISCRETE,
        constants.NEXT_CONTINUOUS,
        constants.NEXT_DISCRETE,
    ]


class TestCreateDataset(object):

    @pytest.mark.parametrize(
        "continuous,discrete,window", [
            (np.arange(20).reshape(5, 4), np.array([]), 2),
            (np.arange(20).reshape(5, 4), np.arange(20).reshape(5, 4), 2),
        ]
    )
    def test_create(self, continuous, discrete, window, dataset_keys):
        actual = dataset.create_dataset(
            pd.DataFrame(continuous),
            pd.DataFrame(discrete),
            window
        )
        assert len(actual) == 2
        batch_size = continuous.shape[1] * window
        a = actual[0]
        for key in dataset_keys:
            if len(discrete) == 0 and key.endswith('discrete'):
                continue
            data = a[key]
            assert data.size == batch_size

    @pytest.mark.parametrize(
        "continuous,discrete,window", [
            (np.arange(20).reshape(2, 10), np.array([]), 2),
            (np.arange(20).reshape(2, 10), np.arange(20).reshape(2, 10), 2),
        ]
    )
    def test_less_length(self, continuous, discrete, window, dataset_keys, caplog):
        actual = dataset.create_dataset(
            pd.DataFrame(continuous),
            pd.DataFrame(discrete),
            window
        )
        warn_recs = 2
        if len(discrete) != 0:
            warn_recs = 4
        assert len(caplog.record_tuples) == warn_recs
        for rec in caplog.record_tuples:
            assert logging.WARNING == rec[1]
        assert len(actual) == 1
        batch_size = continuous.shape[1] * 1
        a = actual[0]
        for key in dataset_keys:
            if len(discrete) == 0 and key.endswith('discrete'):
                continue
            data = a[key]
            assert data.size == batch_size

    @pytest.mark.parametrize(
        "continuous,discrete,window", [
            ([[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]], 1),
            (np.arange(20).reshape(2, 10), np.array([]), 2),
            (np.arange(20).reshape(2, 10), np.arange(20).reshape(2, 10), 2),
        ]
    )
    def test_type_mismatch(self, continuous, discrete, window, dataset_keys, caplog):
        with pytest.raises(AttributeError):
            dataset.create_dataset(continuous, discrete, window)

    def test_assertion(self):
        with pytest.raises(AssertionError):
            dataset.create_dataset([], [], 0)


@pytest.fixture(scope='session')
def dict_dataset():
    return DictDataset(**{
        "test1": [np.arange(10)[-1, None] for _ in range(10)],
        "test2": [np.arange(10)[-1, None] for _ in range(10)],
    })


class TestSplitDataset(object):

    def test_split(self, dict_dataset):
        train, val = dataset.split_dataset(dict_dataset, 0.5)
        assert len(train) == len(val)

    @pytest.mark.parametrize(
        "ratio", [0, 1]
    )
    def test_assertion(self, dict_dataset, ratio):
        with pytest.raises(AssertionError):
            dataset.split_dataset(dict_dataset, ratio)

    def test_empty(self):
        with pytest.raises(ValueError):
            dataset.split_dataset(DictDataset())
