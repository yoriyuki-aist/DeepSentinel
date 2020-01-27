from unittest import mock

import chainer
import numpy as np
import pytest

from deep_sentinel.models.dnn.model.layers import output

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
    "data, output_kinds", [
        (
                # Batch=2, window=2, n_units=2
                [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                3,
        ),
        (
                # Batch=2, window=4, n_units=1
                [[[0], [0], [0], [0]], [[0], [0], [0], [0]]],
                1,
        ),
        (
                # Batch=1, window=2, n_units=3
                [[[0, 0, 0], [0, 0, 0]]],
                2,
        ),
    ]
)
def test_output_layer(data, output_kinds, activate_func, dropout_func):
    given = chainer.Variable(np.array(data).astype(np.float32))
    b, w, n_units = given.shape
    out_layer = output.OutputLayer(n_units, dropout_func, activate_func, output_kinds)
    actual_x, actual_predicted = out_layer(given)
    assert actual_x.shape == (b, w, n_units)
    assert actual_predicted.shape == (b, w, output_kinds)
    assert activate_func.call_count == 1
    assert dropout_func.call_count == 1
