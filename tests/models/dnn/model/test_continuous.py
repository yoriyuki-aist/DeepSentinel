from unittest import mock

import pytest
import numpy as np
import chainer

from chainer import Variable

from deep_sentinel.models.dnn.model import continuous


chainer.global_config.train = False
chainer.global_config.enable_backprop = False

module_path = "deep_sentinel.models.dnn.model.continuous"


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


@pytest.mark.parametrize( "predicted", [
        [
            [1, 0], [1, 0], [1, 0], [1, 0]
        ],
        [
            [1, 0, 1], [1, 0, 2], [1, 0, 4], [1, 0, 3],
        ],
        [
            [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 2], [1, 1, 0, 0, 1, 4], [1, 1, 0, 0, 2, 3],
        ],
    ]
)
def test_sample(predicted):
    predicted_values = to_variable(predicted)
    expected_shape = predicted_values.shape[:-1]
    use_gmm = predicted_values.shape[-1] > 2
    # Dummy sampler
    sampler = mock.MagicMock()
    sampler.sample.return_value = Variable(np.zeros(expected_shape, np.int32))
    with mock.patch("{}.D".format(module_path)) as D_mock:
        D_mock.Categorical.return_value = sampler
        sampled = continuous._sample(predicted_values)
        b, _ = predicted_values.shape
        assert type(sampled) == Variable
        assert sampled.shape == (b, 1)
        assert sampler.sample.call_count == (1 if use_gmm else 0)


class TestContinuousPredictor(object):

    @pytest.mark.parametrize(
        "b,t,h,f,gmm_classes", [
            (1, 1, 1, 1, 1),
            (2, 2, 2, 2, 1),
            (1, 1, 1, 1, 3),
            (2, 2, 2, 2, 3),
        ]
    )
    def test_forward(self, activate_func, dropout_func, b, t, h, f, gmm_classes):
        cp = continuous.ContinuousPredictor(f, h, activate_func, dropout_func, gmm_classes)
        hidden = Variable(np.zeros((b, t, h)).astype(np.float32))
        next_cnt = Variable(np.zeros((b, t, f)).astype(np.float32))
        final_hidden, predicted_set = cp(hidden, next_cnt)
        assert type(final_hidden) == Variable
        assert final_hidden.shape == (b, t, h)
        assert type(predicted_set) == list
        assert len(predicted_set) == f
        for p in predicted_set:
            assert type(p) == Variable
            assert p.shape == (b, t, 2 if gmm_classes == 1 else (3 * gmm_classes))

    @pytest.mark.parametrize(
        "b,h,f,gmm_classes", [
            (1, 1, 1, 1),
            (2, 2, 2, 1),
            (1, 1, 1, 3),
            (2, 2, 2, 3),
        ]
    )
    def test_sample(self, activate_func, dropout_func, b, h, f, gmm_classes):
        cp = continuous.ContinuousPredictor(f, h, activate_func, dropout_func, gmm_classes)
        hidden = Variable(np.zeros((b, h)).astype(np.float32))
        final_hidden, sampled = cp.sample(hidden)
        assert type(final_hidden) == Variable
        assert final_hidden.shape == (b, h)
        assert type(sampled) == Variable
        assert sampled.shape == (b, f, 1)
