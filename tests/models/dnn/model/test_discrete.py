from unittest import mock

import pytest
import numpy as np

import chainer

from chainer import Variable
from deep_sentinel.models.dnn.model import discrete

chainer.global_config.train = False
chainer.global_config.enable_backprop = False


def to_variable(arr, astype=np.float32):
    return Variable(np.array(arr).astype(astype))


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


class TestDiscretePredictor(object):

    @pytest.mark.parametrize(
        "b,t,h,f,gmm_classes,state_kinds", [
            (1, 1, 1, 1, 1, 2),
            (2, 2, 2, 2, 1, 2),
            (1, 1, 1, 1, 3, 2),
            (2, 2, 2, 2, 3, 2),
        ]
    )
    def test_forward(self, b, t, h, f, gmm_classes, state_kinds, activate_func, dropout_func):
        dp = discrete.DiscretePredictor(f, state_kinds, h, activate_func, dropout_func)
        hidden = Variable(np.zeros((b, t, h)).astype(np.float32))
        next_disc = Variable(np.zeros((b, t, f)).astype(np.int32))
        final_hidden, predicted_set = dp(hidden, next_disc)
        assert type(final_hidden) == Variable
        assert final_hidden.shape == (b, t, h)
        assert type(predicted_set) == list
        for p in predicted_set:
            assert type(p) == Variable
            assert p.shape == (b, t, state_kinds)

    @pytest.mark.parametrize(
        "b,h,f,gmm_classes,state_kinds", [
            (1, 1, 1, 1, 2),
            (2, 2, 2, 1, 2),
            (1, 1, 1, 3, 2),
            (2, 2, 2, 3, 2),
        ]
    )
    def test_sample(self, b, h, f, gmm_classes, state_kinds, activate_func, dropout_func):
        dp = discrete.DiscretePredictor(f, state_kinds, h, activate_func, dropout_func)
        hidden = Variable(np.zeros((b * 1, h)).astype(np.float32))
        final_hidden, predicted_values, predicted_dists = dp.sample(hidden)
        assert type(final_hidden) == Variable
        assert final_hidden.shape == (b, h)
        assert type(predicted_values) == Variable
        assert predicted_values.shape == (b, f, 1)
        assert type(predicted_dists) == Variable
        assert predicted_dists.shape == (b, f, state_kinds)
