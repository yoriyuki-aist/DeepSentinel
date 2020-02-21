import pytest

import numpy as np
from chainer import Variable

from deep_sentinel.models.dnn.model.layers import utils


@pytest.mark.parametrize(
    "data,expected,n_batch_axes", [
        (
            [[[0, 1], [0, 1]]],
            [[0, 1], [0, 1]],
            2
        ),
        (
            [[[0], [1]], [[0], [1]]],
            [[0], [1], [0], [1]],
            2
        ),
        (
            [[[0], [1]], [[0], [1]]],
            [[0, 1], [0, 1]],
            1
        ),
        (
            [[0, 1], [0, 1]],
            [[0, 1], [0, 1]],
            2
        )
    ]
)
def test_reshape_batch(data, expected, n_batch_axes):
    given = np.array(data)
    expected = np.array(expected)
    actual = utils.reshape_batch(Variable(given), n_batch_axes)
    assert actual.shape == expected.shape
    np.testing.assert_equal(actual.data, expected)

