from typing import TYPE_CHECKING

from chainer import Chain, ChainList, functions as F, distributions as D

from .layers import mid, output

if TYPE_CHECKING:
    from typing import Callable, List
    from chainer import Variable


def _sample(predicted: 'Variable') -> 'Variable':
    """
    Sampling from Normal distribution.
    :param predicted:        Predicted values
    :return:
    """
    if predicted.shape[-1] > 2:
        # predicted values are the concatenation of mean, scale and weights.
        mean, scale, weights = F.split_axis(predicted, 3, axis=-1)
        # Consider the weights as the probability to which distribution is used.
        weights = F.softmax(weights, axis=-1)
        # Define which distribution to use.
        # TODO: This line is a bottleneck of sampling using GMM feature. Is there any other better way?
        idx = D.Categorical(weights).sample()
        mean = F.select_item(mean, idx)
        scale = F.select_item(scale, idx)
    else:
        mean, scale = F.split_axis(predicted, 2, axis=-1)
    val = F.squeeze(F.gaussian(mean, scale))
    return val.reshape(predicted.shape[0], 1)


class ContinuousPredictor(Chain):
    """Predict the next distribution of continuous values (e.g. sensor values)"""

    def __init__(self, unit_count: int, n_units: int,
                 activation_func: 'Callable[[Variable], Variable]',
                 dropout_func: 'Callable[[Variable], Variable]', gmm_class_count: int = 1):
        """
        Initialization
        :param unit_count:      Number of continuous value units
        :param n_units:         Size of hidden state
        :param activation_func: Activation function to use
        :param dropout_func:    Dropout function to use
        """
        super(ContinuousPredictor, self).__init__()
        self.unit_count = unit_count
        self.n_units = n_units
        self.activation = activation_func
        self.dropout = dropout_func
        mid_layers = ChainList()
        output_layers = ChainList()
        self.gmm_class_count = gmm_class_count
        self.output_units = 2 if self.gmm_class_count == 1 else self.gmm_class_count * 3
        for i in range(self.unit_count - 1):
            mid_layers.append(
                mid.MidLayer(self.n_units, self.dropout, self.activation)
            )
            output_layers.append(
                output.OutputLayer(
                    self.n_units, self.dropout, self.activation,
                    output_kinds=self.output_units)
            )

        with self.init_scope():
            self.first_output_layer = output.OutputLayer(
                self.n_units, self.dropout, self.activation, output_kinds=self.output_units
            )
            self.mid_layers = mid_layers
            self.output_layers = output_layers

    def forward(self, hidden_state: 'Variable', next_value: 'Variable') -> ('Variable', 'List[Variable]'):
        """
        Forward calculation
        :param hidden_state:    Output of `PositionStatePredictor`. The shape is `(B, C, H)`,
                                where `B` is mini batch size, `C` is chunk size, `H` is a size of hidden state.
        :param next_value:      The values at next time step. The shape is `(B, C, M)`, where
                                `M` is number of unit.
        :return:                Final hidden state, and list of predicted mean and variance for each unit.
                                The shape of final hidden state is as same as input hidden state.
                                The shape of list elements is `(B, C, 2)`. First one is mean,
                                and second one is variance. The length of list is as same as `M`.
        """
        hidden_state, predicted = self.first_output_layer(hidden_state)
        predicted_set = [predicted]
        for i in range(self.unit_count - 1):
            mid_output = self.mid_layers[i](next_value[:, :, i:i + 1], hidden_state)
            hidden_state, predicted_values = self.output_layers[i](mid_output)
            predicted_set.append(predicted_values)
        return hidden_state, predicted_set

    def sample(self, hidden_state: 'Variable') -> ('Variable', 'Variable'):
        """
        Exec prediction. Next values are sampled from its distribution.
        :param hidden_state:    Output of LSTM
        :return:                Final hidden state and predicted values, predicted mean and variance
        """
        hidden_state, predicted = self.first_output_layer(hidden_state, 1)
        # Sampling next values
        next_value = _sample(predicted)
        predicted_values = [next_value]
        for i in range(self.unit_count - 1):
            mid_output = self.mid_layers[i](next_value, hidden_state)
            hidden_state, predicted = self.output_layers[i](mid_output, 1)
            next_value = _sample(predicted)
            predicted_values.append(next_value)
        return hidden_state, F.stack(predicted_values, axis=1)

