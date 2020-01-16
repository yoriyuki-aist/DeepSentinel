from typing import TYPE_CHECKING

from chainer import Chain
from chainer import links as L
from chainer import functions as F

if TYPE_CHECKING:
    from chainer import Variable
    from typing import Callable, Tuple


class OutputLayer(Chain):

    def __init__(self, n_units: int, dropout_func: 'Callable[[Variable], Variable]',
                 activation_func: 'Callable[[Variable], Variable]',
                 output_kinds: int = 2):
        super(OutputLayer, self).__init__()
        self.n_units = n_units
        with self.init_scope():
            self.linear = L.Linear(n_units, n_units + output_kinds)
        self.dropout = dropout_func
        self.activation = activation_func

    def forward(self, x: 'Variable', n_batch_axes: int = 2) -> 'Tuple[Variable, Variable]':
        x = self.linear(x, n_batch_axes=n_batch_axes)
        x, predicted = F.split_axis(x, indices_or_sections=[self.n_units], axis=-1)
        x = self.activation(x)
        x = self.dropout(x)
        return x, predicted
