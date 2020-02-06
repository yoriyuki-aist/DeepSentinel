from typing import TYPE_CHECKING


from chainer import Chain, ChainList, functions as F
from .layers import mid, output
from .utils import create_onehot_vector

if TYPE_CHECKING:
    from typing import Callable, List
    from chainer import Variable


def _sample(predicted_probabilities: 'Variable') -> 'Variable':
    """Return most probable values"""
    return F.cast(F.max(predicted_probabilities, axis=-1, keepdims=True), "int")


class DiscretePredictor(Chain):
    """Predict distribution of position state value at next time step (ex. actuator)"""

    def __init__(self, unit_count: int, state_kinds: int, n_units: int,
                 activation_func: 'Callable[[Variable], Variable]',
                 dropout_func: 'Callable[[Variable], Variable]'):
        """
        Initialization
        :param unit_count: Number of units
        :param state_kinds: Number of types of discrete state
        :param n_units: Size of hidden state
        :param activation_func: Activation function to use
        :param dropout_func: Dropout function to use
        """
        super(DiscretePredictor, self).__init__()
        self.n_units = n_units
        self.unit_count = unit_count
        self.activation = activation_func
        self.dropout = dropout_func
        self.state_kinds = state_kinds
        mid_layers = ChainList()
        output_layers = ChainList()
        for i in range(unit_count - 1):
            mid_layers.add_link(
                mid.MidLayer(self.n_units, self.dropout, self.activation, self.state_kinds)
            )
            output_layers.add_link(
                output.OutputLayer(self.n_units, self.dropout, self.activation, self.state_kinds)
            )
        with self.init_scope():
            self.first_output_layer = output.OutputLayer(
                self.n_units, self.dropout, self.activation, self.state_kinds
            )
            self.mid_layers = mid_layers
            self.output_layers = output_layers

    def __call__(self, hidden_state: 'Variable', next_position_values: 'Variable') -> ('Variable', 'List[Variable]'):
        """
        Forward calculation
        :param hidden_state:         Output of each time step of LSTM. The shape is `(B, C, H)`,
                                     where `B` is mini batch size, `C` is chunk size, `H` is size of hidden state.
        :param next_position_values: The position state values at next time step. The shape
                                     is `(B, C, N)`, where `N` is number of unit.
        :return: Final hidden state, and list of probability of state for each unit.
                 The shape of final hidden state is as same as input hidden state. The shape of list
                 elements is `(B, C, K)`, where K is `position_state_kinds`. The length of list is as same as `N`.
        """
        batch_size = next_position_values.shape[0]
        chunk_size = next_position_values.shape[1]
        # Convert state values to One-Hot vector.
        next_pos_onehot_vector = create_onehot_vector(next_position_values, self.state_kinds)
        oh_vec_shape = (batch_size * chunk_size, self.unit_count, self.state_kinds)
        next_pos_onehot_vector = F.reshape(next_pos_onehot_vector, oh_vec_shape)
        hidden_state, predicted = self.first_output_layer(hidden_state)
        predicted_set = [predicted]
        for i in range(self.unit_count - 1):
            mid_output = self.mid_layers[i](next_pos_onehot_vector[:, i], hidden_state)
            hidden_state, predicted_positions = self.output_layers[i](mid_output)
            predicted_set.append(predicted_positions)
        return hidden_state, predicted_set

    def sample(self, hidden_state: 'Variable') -> '(Variable, Variable)':
        """
        Exec prediction. Next values are sampled from its distribution.
        :param hidden_state: Output of PositionStatePredictor
        :return: Final hidden state and predicted values, predicted mean and variance
        """
        batch_size = hidden_state.shape[0]
        hidden_state, predicted = self.first_output_layer(hidden_state, 1)
        # Sampling next values
        next_value = _sample(predicted)
        predicted_distributions = [predicted]
        predicted_values = [next_value]
        for i in range(self.unit_count - 1):
            oh_vec = F.reshape(create_onehot_vector(next_value, self.state_kinds), (batch_size, self.state_kinds))
            mid_output = self.mid_layers[i](oh_vec, hidden_state)
            hidden_state, predicted = self.output_layers[i](mid_output, 1)
            next_value = _sample(predicted)
            predicted_distributions.append(predicted)
            predicted_values.append(next_value)
            # Convert concrete values to one-hot vector
            next_value = create_onehot_vector(next_value, self.state_kinds)
        return hidden_state, F.stack(predicted_values, axis=1), F.stack(predicted_distributions, axis=1)

