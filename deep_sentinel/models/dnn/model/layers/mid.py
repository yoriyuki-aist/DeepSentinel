from typing import TYPE_CHECKING

from chainer import Chain
from chainer import functions as F
from .batch_linear import BatchBiLinear

if TYPE_CHECKING:
    from chainer import Variable
    from typing import Callable


class MidLayer(Chain):

    def __init__(self, n_units: int, dropout_func: 'Callable[[Variable], Variable]',
                 activation_func: 'Callable[[Variable], Variable]', feature_kinds: int = 1):
        super(MidLayer, self).__init__()
        self.feature_kinds = feature_kinds
        with self.init_scope():
            self.bi_linear = BatchBiLinear(feature_kinds, n_units, n_units)
        self.dropout = dropout_func
        self.activation = activation_func

    def forward(self, value: 'Variable', hidden_state: 'Variable', n_batch_axes: int = 2) -> 'Variable':
        """
        Mid output of DeepSentinel
        :param value:           the value of next time step.
                                The shape must be `(batch, window, features)`.
        :param hidden_state:    Previous layer's output. The shape must be `(batch, window, n_units)`.
        :param n_batch_axes:    Index of mini batch axes
        :return:
        """
        batch_size = hidden_state.shape[0]
        chunk_size = hidden_state.shape[1]
        h = self.bi_linear(value, hidden_state, n_batch_axes)
        h = self.activation(h)
        h = self.dropout(h)
        # The shape is `(B*C, H)`. Convert it to `(B, C, H)`
        return F.reshape(h, (batch_size, chunk_size, -1))
