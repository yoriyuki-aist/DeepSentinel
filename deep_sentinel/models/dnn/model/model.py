from typing import TYPE_CHECKING

from chainer import Chain, Sequential, links as L, functions as F

from .continuous import ContinuousPredictor
from .discrete import DiscretePredictor
from .utils import ActivationFunc, get_dropout_func, create_onehot_vector

if TYPE_CHECKING:
    from chainer import Variable


class DeepSentinel(Chain):

    def __init__(self, continuous_unit_count: int, discrete_unit_count: int, discrete_state_kinds: int, n_units: int,
                 activation_func_type: str = 'sigmoid', dropout_ratio: float = 0.5, lstm_stack: int = 1,
                 gmm_classes: int = 1):
        """
        Initialization
        :param continuous_unit_count: Number of continuous value units
        :param discrete_unit_count: Number of discrete value units
        :param discrete_state_kinds: Number of types of discrete
        :param n_units: Size of hidden state
        :param activation_func_type: 'sigmoid' or 'relu'
        :param dropout_ratio: Dropout ratio. Default is 0.5
        :param gmm_classes: Number of Gaussian distribution to use for GMM
        """
        assert activation_func_type.lower() in ActivationFunc.choices(), \
            "`{}` is not supported.".format(activation_func_type)
        assert 0 <= dropout_ratio < 1, "`dropout_ratio` must be 0 <= `dropout_ratio` < 1."
        super(DeepSentinel, self).__init__()
        self.discrete_unit_count = discrete_unit_count
        self.continuous_unit_count = continuous_unit_count
        self.discrete_state_kinds = discrete_state_kinds
        self.activation = ActivationFunc[activation_func_type.upper()]
        self.dropout = get_dropout_func(dropout_ratio)
        self.gmm_classes = gmm_classes
        with self.init_scope():
            self.input_layer = Sequential(
                L.Linear(
                    in_size=discrete_state_kinds * discrete_unit_count + continuous_unit_count,
                    out_size=n_units
                ),
                self.dropout,
            )
            self.lstm = L.NStepLSTM(
                n_layers=lstm_stack,
                in_size=n_units,
                out_size=n_units,
                dropout=dropout_ratio
            )
            self.discrete_predictor = DiscretePredictor(
                unit_count=discrete_unit_count,
                state_kinds=discrete_state_kinds,
                n_units=n_units,
                activation_func=self.activation,
                dropout_func=self.dropout
            )
            self.continuous_predictor = ContinuousPredictor(
                unit_count=continuous_unit_count,
                n_units=n_units,
                activation_func=self.activation,
                dropout_func=self.dropout,
                gmm_class_count=gmm_classes
            )

        self.prev_hidden_state = None
        self.prev_cell_state = None

    def _convert_to_input(self, current_discrete_values: 'Variable', current_continuous_values: 'Variable'):
        # Obtain one-hot vector
        default_shape = current_discrete_values.shape
        target_shape = (default_shape[0], default_shape[1], default_shape[2] * self.discrete_state_kinds)
        current_discrete_values = create_onehot_vector(current_discrete_values, self.discrete_state_kinds)
        current_discrete_values = F.reshape(current_discrete_values, target_shape)
        # Concatenated matrix is as input
        return F.concat([current_discrete_values, current_continuous_values], axis=-1)

    def forward(self, current_continuous_values: 'Variable', next_continuous_values: 'Variable',
                current_discrete_values: 'Variable', next_discrete_values: 'Variable') -> '(Variable, Variable)':
        """
        Forward calculation
        :param current_continuous_values:   Continuous values of current record
        :param current_discrete_values:     Discrete values of current record
        :param next_continuous_values:      Continuous values of next record
        :param next_discrete_values:        Discrete values of next record
        :return: Predicted mean and variance for each unit, and predicted discrete vectors for each unit
        """
        inputs = self._convert_to_input(current_discrete_values, current_continuous_values)
        hidden_state = self.input_layer(inputs, 2)
        self.prev_hidden_state, self.prev_cell_state, hidden_state_per_step = self.lstm(
            self.prev_hidden_state, self.prev_cell_state, F.separate(hidden_state)
        )
        hidden_state, predicted_discrete_states = self.discrete_predictor(
            F.stack(hidden_state_per_step), next_discrete_values
        )
        _, predicted_values = self.continuous_predictor(
            hidden_state, next_continuous_values
        )
        return F.stack(predicted_values, 2), F.stack(predicted_discrete_states, 2)

    def initialize_with(self, initial_continuous_values: 'Variable',
                        initial_discrete_values: 'Variable') -> 'DeepSentinel':
        """
        :param initial_continuous_values:   Initialize states of LSTM with this values. The shape must be
                                            `(B, T, K)`, where `B` means batch size, `T` means time range,
                                             and `K` means the number of unit to predict. `K` is as same as 
                                            `self.continuous_unit_count`.
        :param initial_discrete_values:     Initialize states of LSTM with this value. The shape must be
                                            `(B, T, P)`, where `P` means the number of unit of discrete values.
                                            `P` is as same as `self.discrete_unit_count`.
        :return:
        """
        inputs = self._convert_to_input(initial_discrete_values, initial_continuous_values)
        # Initialize states
        hidden_state = self.input_layer(inputs, 2)
        self.prev_hidden_state, self.prev_cell_state, _ = self.lstm(
            self.prev_hidden_state, self.prev_cell_state, F.separate(hidden_state)
        )
        return self

    def sample(self, steps: int) -> '(Variable, Variable, Variable, Variable)':
        """
        Execute multi step prediction. Use initial values to initialize the state of model.
        Then, start prediction until specified step. This method do not require the values
        at next time step.
        :param steps:   Number of steps to predict
        :return: Predicted values, and predicted distribution.
        """
        assert self.prev_hidden_state is not None, "YOu should call `initialize_with()` before sampling"
        position_values_set = list()
        position_probability_set = list()
        continuous_values_set = list()
        continuous_distribution_set = list()
        for i in range(steps):
            # Reshape to `(B, L, K)` from `(L, B, K)`. `L` means LSTM stack (always 1 here).
            hidden_state = F.swapaxes(self.prev_hidden_state[-1:], 0, 1)
            hidden_state, predicted_values, predicted_dists = self.discrete_predictor.sample(hidden_state)
            position_values_set.append(predicted_values)
            position_probability_set.append(predicted_dists)
            hidden_state, predicted_values, predicted_dists = self.continuous_predictor.sample(hidden_state)
            continuous_values_set.append(predicted_values)
            continuous_distribution_set.append(predicted_dists)
            inputs = self._convert_to_input(position_values_set[-1], continuous_values_set[-1])
            self.prev_hidden_state, self.prev_cell_state, _ = self.lstm(
                self.prev_hidden_state, self.prev_cell_state, F.separate(self.input_layer(inputs, 2))
            )
        return (
            F.hstack(continuous_values_set), F.hstack(continuous_distribution_set),
            F.hstack(position_values_set), F.hstack(position_probability_set)
        )

    def reset_state(self):
        """Reset the states of NStepLSTM"""
        self.prev_hidden_state = None
        self.prev_cell_state = None


class DeepSentinelWithoutDiscrete(Chain):

    def __init__(self, continuous_unit_count: int, n_units: int, activation_func_type: str = 'sigmoid',
                 dropout_ratio: float = 0.5, lstm_stack: int = 1, gmm_class_count: int = 1):
        """
        Initialization
        :param continuous_unit_count: Number of continuous value units
        :param n_units: Size of hidden state
        :param activation_func_type: 'sigmoid' or 'relu'
        :param dropout_ratio: Dropout ratio. Default is 0.5
        """
        assert activation_func_type.lower() in ActivationFunc.choices(), \
            "`{}` is not supported.".format(activation_func_type)
        assert 0 <= dropout_ratio < 1, "`dropout_ratio` must be 0 <= `dropout_ratio` < 1."
        super(DeepSentinelWithoutDiscrete, self).__init__()
        self.continuous_unit_count = continuous_unit_count
        self.activation = ActivationFunc[activation_func_type.upper()]
        self.dropout = get_dropout_func(dropout_ratio)
        self.gmm_class_count = gmm_class_count
        self.output_units = 2 if self.gmm_class_count == 1 else 3
        with self.init_scope():
            self.input_layer = Sequential(
                L.Linear(in_size=continuous_unit_count, out_size=n_units),
                self.dropout,
            )
            self.lstm = L.NStepLSTM(
                n_layers=lstm_stack,
                in_size=n_units,
                out_size=n_units,
                dropout=dropout_ratio
            )
            self.continuous_predictor = ContinuousPredictor(
                unit_count=continuous_unit_count,
                n_units=n_units,
                activation_func=self.activation,
                dropout_func=self.dropout,
                gmm_class_count=self.gmm_class_count
            )

        self.prev_hidden_state = None
        self.prev_cell_state = None

    def forward(self, current_continuous_values: 'Variable', next_continuous_values: 'Variable') -> '(Variable, None)':
        """
        Forward calculation
        :param current_continuous_values:   Continuous values of current record
        :param next_continuous_values:      Continuous values of next record
        :return: Predicted mean and variance for each unit
        """
        # Concatenated matrix is as input
        hidden_state = self.input_layer(current_continuous_values, 2)
        self.prev_hidden_state, self.prev_cell_state, hidden_state_per_step = self.lstm(
            self.prev_hidden_state, self.prev_cell_state, F.separate(hidden_state)
        )
        # Execute prediction
        _, predicted_values = self.continuous_predictor(F.stack(hidden_state_per_step), next_continuous_values)
        return F.stack(predicted_values, 2), None

    def initialize_with(self, initial_values: 'Variable') -> 'Variable':
        """
        :param initial_values: Initialize states of LSTM with this values. The shape must be
                               `(B, T, K)`, where `B` means batch size, `T` means time range,
                                and `K` means the number of unit to predict. `K` is as same as
                                `self.continuous_unit_count`.
        :return:
        """
        self.reset_state()
        hidden_state = self.input_layer(initial_values, 2)
        self.prev_hidden_state, self.prev_cell_state, _ = self.lstm(
            self.prev_hidden_state, self.prev_cell_state, F.separate(hidden_state)
        )
        return hidden_state

    def sample(self, steps: int) -> '(Variable, Variable)':
        """
        Execute multi step prediction. Use initial values to initialize the state of model.
        Then, start prediction until specified step. This method do not require the values
        at next time step.
        :param steps:   Number of steps to predict
        :return: Predicted values, predicted mean and std.
        """
        batch_size = self.prev_hidden_state.shape[1]
        values_set = list()
        for i in range(steps):
            # Reshape to `(B, L, K)` from `(L, B, K)`. `L` means LSTM stack (always 1 here).
            hidden_state = self.prev_hidden_state[-1]
            hidden_state, predicted_values = self.continuous_predictor.sample(hidden_state)
            input_val = self.input_layer(predicted_values, 1)
            self.prev_hidden_state, self.prev_cell_state, _ = self.lstm(
                self.prev_hidden_state, self.prev_cell_state,
                F.split_axis(input_val, input_val.shape[0], axis=0)
            )
            values_set.append(predicted_values)
        values_set = F.stack(values_set, axis=1).reshape(batch_size, steps, self.continuous_unit_count)
        return values_set

    def reset_state(self):
        """Reset the states of NStepLSTM"""
        self.prev_hidden_state = None
        self.prev_cell_state = None
