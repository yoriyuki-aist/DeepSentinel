from unittest import mock

import chainer
import numpy as np
import pytest

from deep_sentinel.models.dnn.model.layers import mid

chainer.global_config.train = False
chainer.global_config.enable_backprop = False


@pytest.fixture
def activate_func():
    m = mock.MagicMock()
    m.side_effect = lambda x: x
    return m


@pytest.fixture
def dropout_func():
    m = mock.MagicMock()
    m.side_effect = lambda x: x
    return m


@pytest.mark.parametrize(
    "data, n_units", [
        (
                # Batch=2, window=2, feature=2
                [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                5,
        ),
        (
                # Batch=2, window=4, feature=1
                [[[0], [0], [0], [0]], [[0], [0], [0], [0]]],
                4,
        ),
        (
                # Batch=1, window=2, feature=3
                [[[0, 0, 0], [0, 0, 0]]],
                3,
        ),
    ]
)
def test_mid_layer(data, n_units, activate_func, dropout_func):
    given = chainer.Variable(np.array(data).astype(np.float32))
    b, w, f = given.shape
    hidden = chainer.Variable(
        np.arange(b * w * n_units)
            .reshape((b, w, n_units))
            .astype(np.float32)
    )
    mid_layer = mid.MidLayer(n_units, dropout_func, activate_func, f)
    actual = mid_layer(given, hidden)
    assert actual.shape == (b, w, n_units)
    assert activate_func.call_count == 1
    assert dropout_func.call_count == 1
