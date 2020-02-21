import itertools


import pytest
import numpy as np

import chainer
from chainer import Variable

from deep_sentinel.models.dnn.model import model

chainer.global_config.train = False
chainer.global_config.enable_backprop = False
np.random.seed(0)


def zero_variables(shape, astype=np.float32):
    return Variable(np.zeros(shape, astype))


class TestDeepSentinel(object):

    @pytest.mark.parametrize(
        "activ_func,dropout,exc", [
            ('other', 0.5, AssertionError),
            ('sigmoid', 1, AssertionError),
            ('relu', -1, AssertionError),
        ]
    )
    def test_init_err(self, activ_func, dropout, exc):
        with pytest.raises(exc):
            model.DeepSentinel(3, 3, 3, 3, activ_func, dropout)

    @pytest.mark.parametrize(
        "b, t, c, d, kinds", [
            (1, 1, 1, 1, 3),
            (2, 2, 2, 2, 3),
        ]
    )
    def test_convert_to_input(self, b, t, c, d, kinds):
        ds = model.DeepSentinel(c, d, kinds, 3, "sigmoid", 0.3)
        cnt = Variable(np.zeros((b, t, c), np.float32))
        disc = Variable(np.zeros((b, t, d), np.int32))
        actual = ds._convert_to_input(disc, cnt)
        assert type(actual) == Variable
        assert actual.shape == (b, t, c + d * kinds)

    @pytest.mark.parametrize(
        "b, t, c, d, n, l, g",
        list(itertools.product(
            [1, 2],  # batch size
            [1, 2],  # time length
            [1, 2],  # continuous units
            [1, 2],  # discrete units
            [1, 2],  # n units
            [1, 2],  # lstm stack
            [1, 2],  # GMM classes
        ))
    )
    def test_forward(self, b, t, c, d, n, l, g):
        ds = model.DeepSentinel(c, d, 3, n, 'sigmoid', 0.5, l, g)
        cur_cnt = zero_variables((b, t, c))
        next_cnt = zero_variables((b, t, c))
        cur_disc = zero_variables((b, t, d), np.int32)
        next_disc = zero_variables((b, t, d), np.int32)
        predicted = ds(cur_cnt, next_cnt, cur_disc, next_disc)
        assert type(predicted) == tuple
        assert len(predicted) == 2
        p_values = predicted[0]
        p_states = predicted[1]
        assert type(p_values) == Variable
        assert type(p_states) == Variable
        assert p_values.shape == (b, t, c, 2 if g == 1 else 3 * g)
        assert p_states.shape == (b, t, d, 3)

    @pytest.mark.parametrize(
        "b, t, c, d, n, l, g, steps",
        list(itertools.product(
            [1, 9],  # batch size
            [1, 8],  # time length
            [1, 7],  # continuous units
            [1, 6],  # discrete units
            [1, 5],  # n units
            [1, 4],  # lstm stack
            [1, 3],  # GMM classes
            [1, 2],  # Steps
        ))
    )
    def test_sample(self, b, t, c, d, n, l, g, steps):
        ds = model.DeepSentinel(c, d, 3, n, 'sigmoid', 0.5, l, g)
        ds.prev_hidden_state = zero_variables((l, b, n))
        ds.prev_cell_state = zero_variables((l, b, n))
        sampled = ds.sample(steps)
        assert type(sampled) == tuple
        assert len(sampled) == 2
        p_values = sampled[0]
        p_states = sampled[1]
        assert type(p_values) == Variable
        assert type(p_states) == Variable
        assert p_values.shape == (b, steps, c)
        assert p_states.shape == (b, steps, d)

    @pytest.mark.parametrize(
        "b, t, c, d, kinds, lstm_stack", [
            (1, 1, 1, 1, 3, 1),
            (1, 1, 1, 1, 3, 2),
            (2, 2, 2, 2, 3, 1),
            (2, 2, 2, 2, 3, 2),
        ]
    )
    def test_initialize_with(self, b, t, c, d, kinds, lstm_stack):
        init_cnt = zero_variables((b, t, c))
        init_disc = zero_variables((b, t, d), np.int32)
        ds = model.DeepSentinel(c, d, kinds, 3, "sigmoid", 0.3, lstm_stack=lstm_stack)
        m = ds.initialize_with(init_cnt, init_disc)
        assert type(m) == model.DeepSentinel
        assert ds.prev_hidden_state is not None
        assert ds.prev_cell_state is not None
        assert type(ds.prev_hidden_state) == Variable
        assert type(ds.prev_cell_state) == Variable
        assert ds.prev_hidden_state.shape == (lstm_stack, b, 3)
        assert ds.prev_cell_state.shape == (lstm_stack, b, 3)

    def test_reset_state(self):
        hidden = zero_variables((1, 1, 1))
        ds = model.DeepSentinel(1, 1, 3, 3, "sigmoid", 0.3)
        ds.prev_hidden_state = hidden
        ds.prev_cell_state = hidden
        assert ds.prev_hidden_state is not None
        assert ds.prev_cell_state is not None
        ds.reset_state()
        assert ds.prev_hidden_state is None
        assert ds.prev_cell_state is None


class TestDeepSentinelWithoutDiscrete(object):

    @pytest.mark.parametrize(
        "activ_func,dropout,exc", [
            ('other', 0.5, AssertionError),
            ('sigmoid', 1, AssertionError),
            ('relu', -1, AssertionError),
        ]
    )
    def test_init_err(self, activ_func, dropout, exc):
        with pytest.raises(exc):
            model.DeepSentinelWithoutDiscrete(3, 3, activ_func, dropout)

    @pytest.mark.parametrize(
        "b, t, c, n, l, g",
        list(itertools.product(
            [1, 2],  # batch size
            [1, 2],  # time length
            [1, 2],  # continuous units
            [1, 2],  # n units
            [1, 2],  # lstm stack
            [1, 2],  # GMM classes
        ))
    )
    def test_forward(self, b, t, c, n, l, g):
        ds = model.DeepSentinelWithoutDiscrete(c, n, 'sigmoid', 0.5, l, g)
        cur_cnt = zero_variables((b, t, c))
        next_cnt = zero_variables((b, t, c))
        predicted = ds(cur_cnt, next_cnt)
        assert type(predicted) == tuple
        assert len(predicted) == 2
        p_values = predicted[0]
        p_states = predicted[1]
        assert type(p_values) == Variable
        assert p_values.shape == (b, t, c, 2 if g == 1 else g * 3)
        assert p_states is None

    @pytest.mark.parametrize(
        "b, t, c, n, l, g, steps",
        list(itertools.product(
            [1, 9],  # batch size
            [1, 8],  # time length
            [1, 7],  # continuous units
            [1, 5],  # n units
            [1, 4],  # lstm stack
            [1, 3],  # GMM classes
            [1, 2],  # Steps
        ))
    )
    def test_sample(self, b, t, c, n, l, g, steps):
        dswd = model.DeepSentinelWithoutDiscrete(c, n, 'sigmoid', 0.5, l, g)
        dswd.prev_hidden_state = zero_variables((l, b, n))
        dswd.prev_cell_state = zero_variables((l, b, n))
        sampled = dswd.sample(steps)
        assert type(sampled) == tuple
        assert len(sampled) == 2
        p_values = sampled[0]
        p_states = sampled[1]
        assert type(p_values) == Variable
        assert p_states is None
        assert p_values.shape == (b, steps, c)
