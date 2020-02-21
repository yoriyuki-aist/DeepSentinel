from unittest.mock import MagicMock, patch, call

import pytest
import numpy as np

import chainer
from chainer import Variable

from deep_sentinel.models.dnn.model import loss_calculator

chainer.global_config.train = False
chainer.global_config.enable_backprop = False


def to_variable(arr, astype=np.float32):
    return Variable(np.array(arr).astype(astype))


@pytest.mark.parametrize(
    "predicted,actual", [
        ([[[[1, 0]], [[1, 0]]], [[[1, 0]], [[1, 0]]]],
         [[[1], [1]], [[1], [1]]]),
        ([[[[1, 0], [0, 1]], [[1, 0], [0, 1]]], [[[1, 0], [1, 0]], [[1, 0], [0, 1]]]],
         [[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    ]
)
def test_continuous_loss(predicted, actual):
    # (batch, time length, features, 2)
    # [[[[mean, scale], [...]]], ...]
    predicted_values = to_variable(predicted)
    # (batch, time length, features)
    actual_values = to_variable(actual)
    loss = loss_calculator.calculate_continuous_value_loss(predicted_values, actual_values)
    # (batch, time length, features)
    assert type(loss) == Variable
    assert loss.shape == actual_values.shape


@pytest.mark.parametrize(
    "predicted,actual", [
        ([[[[0, 0, 1]], [[0, 0, 1]]], [[[1, 0, 0]], [[1, 0, 0]]]],
         [[[1], [2]], [[2], [2]]]),
        ([[[[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]]], [[[1, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 1]]]],
         [[[1, 1], [2, 2]], [[2, 1], [2, 1]]]),
    ]
)
def test_discrete_loss(predicted, actual):
    # (batch, time length, features, state kinds)
    predicted_values = to_variable(predicted)
    # (batch, time length, features)
    actual_values = to_variable(actual, astype=np.int32)
    loss = loss_calculator.calculate_discrete_value_loss(predicted_values, actual_values)
    # (batch, time length, features)
    assert type(loss) == Variable
    assert loss.shape == actual_values.shape


@pytest.mark.parametrize(
    "predicted,actual", [
        ([[[[1, 1, 0, 0, 1, 1]], [[1, 1, 0, 0, 1, 2]]], [[[1, 1, 0, 0, 1, 4]], [[1, 1, 0, 0, 2, 3]]]],
         [[[1], [1]], [[1], [1]]]),
        ([[[[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 2, 1]], [[1, 1, 0, 0, 1, 2], [0, 0, 1, 1, 2, 1]]], [[[1, 1, 0, 0, 9, 1], [1, 1, 0, 0, 2, 3]], [[1, 1, 0, 0, 2, 0], [0, 0, 1, 1, 2, 1]]]],
         [[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    ]
)
def test_gmm_loss(predicted, actual):
    # (batch, time length, features, gmm classes * 3)
    # [[[[(mean1, mean2, std1, std2, weight1, weight2)]], [...]]]
    predicted_values = to_variable(predicted)
    # (batch, time length, features)
    actual_values = to_variable(actual)
    loss = loss_calculator.calculate_gmm_nll(predicted_values, actual_values)
    # (batch, time length, features)
    assert type(loss) == Variable
    assert loss.shape == actual_values.shape


class TestLossCalculator(object):

    module_path = "deep_sentinel.models.dnn.model.loss_calculator"

    def test_set_as_predict(self):
        lc = loss_calculator.LossCalculator(None)
        assert lc.is_training
        lc.set_as_predict()
        assert not lc.is_training

    @pytest.mark.parametrize(
        "cur_disc,next_disc,is_training", [
            ([0], [1], False),
            ([0], [1], True),
            ([0], None, False),
            ([0], None, True),
            (None, [1], False),
            (None, [1], True),
            (None, None, False),
            (None, None, True),
        ]
    )
    def test__call__(self, cur_disc, next_disc, is_training):
        cur_cnt = to_variable([1])
        next_cnt = to_variable([1])
        cur_disc = cur_disc if cur_disc is None else to_variable(cur_disc, np.int32)
        next_disc = next_disc if next_disc is None else to_variable(next_disc, np.int32)
        # Dummy model
        dummy_loss1 = to_variable([1])
        dummy_loss2 = to_variable([1])
        model = MagicMock()
        model.return_value = (dummy_loss1, dummy_loss2)
        # Create an instance of LossCalculator
        lc = loss_calculator.LossCalculator(model)
        lc.is_training = is_training
        with patch("{}.calculate_continuous_value_loss".format(self.module_path)) as c_loss:
            with patch("{}.calculate_discrete_value_loss".format(self.module_path)) as d_loss:
                c_loss.return_value = to_variable([1, 0])
                d_loss.return_value = to_variable([0, 1])
                loss = lc(cur_cnt, next_cnt, cur_disc, next_disc)
                # Assert called functions
                c_loss.assert_has_calls([call(dummy_loss1, next_cnt)])
                if cur_disc is not None and next_disc is not None:
                    model.assert_has_calls([call(cur_cnt, next_cnt, cur_disc, next_disc)])
                    d_loss.assert_has_calls([call(dummy_loss2, next_disc)])
                else:
                    model.assert_has_calls([call(cur_cnt, next_cnt)])
                    d_loss.assert_not_called()
                assert type(loss) == Variable
                assert c_loss.call_count == 1
                if is_training:
                    assert loss.size == 1
                else:
                    if cur_disc is not None and next_disc is not None:
                        assert loss.shape == (4,)
                    else:
                        assert loss.shape == (2,)

    @pytest.mark.parametrize(
        "cnt,disc", [
            (to_variable([0]), None),
            (to_variable([0]), to_variable([1])),
        ]
    )
    def test_initialize_with(self, cnt, disc):
        model = MagicMock()
        model.initialize_with.return_value = "initialized"
        lc = loss_calculator.LossCalculator(model)
        assert lc.initialize_with(cnt, disc) == "initialized"
        if disc is None:
            model.initialize_with.assert_has_calls([call(cnt)])
        else:
            model.initialize_with.assert_has_calls([call(cnt, disc)])

    def test_sample(self):
        model = MagicMock()
        model.sample.return_value = "sampled"
        lc = loss_calculator.LossCalculator(model)
        assert lc.sample(1) == "sampled"
        model.sample.assert_has_calls([call(1)])

    def test_reset_state(self):
        model = MagicMock()
        lc = loss_calculator.LossCalculator(model)
        lc.reset_state()
        model.reset_state.assert_has_calls([call()])
