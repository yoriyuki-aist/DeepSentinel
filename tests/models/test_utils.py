from unittest import mock

import pytest

import numpy as np

from chainer import Variable
from deep_sentinel.models.dnn.model import utils


class TestActivationFunc(object):

    @pytest.mark.parametrize(
        "func_name", [
            "sigmoid",
            "relu"
        ]
    )
    def test_call(self, func_name):
        with mock.patch('deep_sentinel.models.dnn.model.utils.F.{}'.format(func_name),
                        return_value=func_name) as m:
            activate_func = getattr(utils.ActivationFunc, func_name.upper())
            actual = activate_func(0)
            assert m.call_count == 1
            assert m.call_args[0] == (0,)
            assert actual == func_name

    def test_not_supported(self):
        with pytest.raises(AttributeError):
            getattr(utils.ActivationFunc, "NOT_SUPPORTED")

    def test_supported_func(self):
        expected = ["sigmoid", "relu"]
        actual = utils.ActivationFunc.choices()
        assert set(actual) == set(expected)


class TestGetDropoutFunc(object):

    def test_call(self):
        with mock.patch('deep_sentinel.models.dnn.model.utils.F.dropout',
                        return_value=0) as m:
            dropout = utils.get_dropout_func(0.5)
            actual = dropout(0)
            assert m.call_count == 1
            assert m.call_args == ((0,), {'ratio': 0.5})
            assert actual == 0

    @pytest.mark.parametrize(
        "ratio", [
            -1, 1
        ]
    )
    def test_error(self, ratio):
        with pytest.raises(AssertionError):
            utils.get_dropout_func(ratio)


class TestCreateOneHotVector(object):

    @pytest.mark.parametrize(
        "data,n_types", [
            ([1, 2, 3], 4),
            ([[1], [2], [3]], 4),
        ]
    )
    def test_create(self, data, n_types):
        data = np.array(data)
        actual = utils.create_onehot_vector(Variable(data), n_types)
        assert len(actual) == len(data)
        assert actual.size == len(actual) * n_types

    @pytest.mark.parametrize(
        "data,n_types,exc", [
            ([1], 0, AssertionError),
            ([[1], [2], [3]], 2, IndexError),
        ]
    )
    def test_error(self, data, n_types, exc):
        data = np.array(data)
        with pytest.raises(exc):
            utils.create_onehot_vector(Variable(data), n_types)

