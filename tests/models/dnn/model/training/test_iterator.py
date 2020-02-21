import math

import pytest
import numpy as np

from chainer import Variable
from chainer.datasets import DictDataset
from chainer.dataset import concat_examples

from deep_sentinel.models.dnn.training import iterator


@pytest.fixture(scope="module")
def normal_dataset():
    arr = np.arange(10000).reshape(1000, 10).astype(np.float32)
    return DictDataset(data=Variable(arr))


class TestParallelSequentialIterator(object):

    @pytest.mark.parametrize(
        "batchsize", [
            1,
            15,
            100,
            1000,
        ]
    )
    def test_iter(self, batchsize, normal_dataset):
        psi = iterator.ParallelSequentialIterator(normal_dataset, batchsize, repeat=False)
        n = 0
        for i in psi:
            data = concat_examples(i)
            assert "data" in data
            assert data["data"].shape == (batchsize, 10)
            n += 1
        assert n == math.ceil(1000 / batchsize)

    def test_overiter(self, normal_dataset):
        batchsize = 10000
        with pytest.raises(AssertionError):
            iterator.ParallelSequentialIterator(normal_dataset, batchsize, repeat=False)

    def test_repeatable(self, normal_dataset):
        batchsize = 1
        psi = iterator.ParallelSequentialIterator(normal_dataset, batchsize, repeat=True)
        first = psi.next()[0]
        for i in range((1000 // batchsize) - 1):
            psi.next()
        re_first = psi.next()[0]
        assert first["data"].shape == (10,)
        assert re_first["data"].shape == (10,)
        assert (first["data"].array == re_first["data"].array).all()


class TestMultipleSequenceIterator(object):

    @pytest.mark.parametrize(
        "batch_ratio", [
            1, 2, 3
        ]
    )
    def test_iter(self, batch_ratio, normal_dataset):
        msi = iterator.MultipleSequenceIterator([normal_dataset], batch_ratio=batch_ratio, repeat=False)
        for i in msi:
            assert len(i) == batch_ratio

    @pytest.mark.parametrize(
        "batch_ratio,exc", [
            (0, ValueError),
            (-1, ValueError),
            (0.1, TypeError)
        ]
    )
    def test_ratio_error(self, normal_dataset, batch_ratio, exc):
        with pytest.raises(exc):
            iterator.MultipleSequenceIterator([normal_dataset], batch_ratio)

    def test_value_error(self, normal_dataset):
        arr = np.arange(100).reshape(10, 10).astype(np.float32)
        another_dataset = DictDataset(data=Variable(arr))
        with pytest.raises(ValueError):
            iterator.MultipleSequenceIterator([normal_dataset, another_dataset])

