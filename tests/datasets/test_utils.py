import pytest
import numpy as np
import pandas as pd

from deep_sentinel.dataset import utils


@pytest.mark.parametrize(
    "continuous,discrete", [
        ({'a': [1.0, 2.0, 3.0]}, {}),
        ({'a': [1.0, 2.0, 3.0]}, {'b': [1, 2, 3]}),
        ({}, {'b': [1, 2, 3]}),
    ]
)
def test_split_category_data(continuous, discrete):
    discrete_keys = list(discrete.keys())
    continuous_keys = list(continuous.keys())
    given = pd.concat(
        [
            pd.DataFrame(continuous) if len(continuous) != 0 else pd.DataFrame(),
            pd.DataFrame(discrete).astype("category") if len(discrete) != 0 else pd.DataFrame(),
        ], axis=1
    )
    actual = utils.split_category_data(given)
    pd.testing.assert_frame_equal(actual[0], given[continuous_keys])
    pd.testing.assert_frame_equal(actual[1], given[discrete_keys])


class TestSplitNdarray(object):

    @pytest.mark.parametrize(
        "given,ratio,expected", [
            ([0, 1, 2, 3], 0.5, [[0, 1], [2, 3]]),
            ([0, 1, 2, 3], 0.2, [[0], [1, 2, 3]]),
            ([[0], [1], [2], [3]], 0.5, [[[0], [1]], [[2], [3]]]),
        ]
    )
    def test_split(self, given, ratio, expected):
        expected = np.array(expected)
        actual = utils.split_ndarray(np.array(given), ratio)
        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "given,ratio,err_type", [
            ([0, 1, 2, 3], 0.0, AssertionError),
            ([0, 1, 2, 3], 1.0, AssertionError),
        ]
    )
    def test_split(self, given, ratio, err_type):
        with pytest.raises(err_type):
            utils.split_ndarray(np.array(given), ratio)
