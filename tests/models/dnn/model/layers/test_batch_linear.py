import pytest

import numpy as np
import chainer

from deep_sentinel.models.dnn.model.layers import batch_linear

chainer.global_config.train = False
chainer.global_config.enable_backprop = False


@pytest.mark.parametrize(
    "data,expected,n_batch_axes,args", [
        (
            [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            2,
            (2, 2, 2)
        ),
        (
            [[[0, 0]], [[0, 0]]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            2,
            (2, 2, 4)
        )
    ]
)
def test_batch_bi_linear(data, expected, n_batch_axes, args):
    given = chainer.Variable(np.array(data).astype(np.float32))
    expected = chainer.Variable(np.array(expected).astype(np.float32))
    actual = batch_linear.BatchBiLinear(*args)(given, given.T, n_batch_axes)
    assert actual.shape == expected.shape
    assert (actual.array == expected.array).all()
