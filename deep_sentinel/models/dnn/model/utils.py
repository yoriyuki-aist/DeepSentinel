import functools
from enum import Enum
from typing import TYPE_CHECKING

from chainer import functions as F
from chainer.backend import get_array_module, cuda


if TYPE_CHECKING:
    from typing import Callable
    from chainer import Variable


class ActivationFunc(Enum):
    SIGMOID = 0
    RELU = 1

    @classmethod
    def choices(cls):
        return tuple(t.name.lower() for t in cls)

    def __call__(self, *args, **kwargs):
        if self.name == self.SIGMOID.name:
            return F.sigmoid(*args, **kwargs)
        elif self.name == self.RELU.name:
            return F.relu(*args, **kwargs)


def get_dropout_func(dropout_ratio: 0.5) -> 'Callable[[Variable], Variable]':
    """Obtain dropout function"""
    dropout_func = functools.partial(F.dropout, ratio=dropout_ratio)
    # noinspection PyTypeChecker
    return dropout_func  # NOQA


def create_onehot_vector(x: 'Variable', number_of_types: int) -> 'Variable':
    """
    Create one-hot vector.
    :param x: Variable to convert
    :param number_of_types: Number of types of values
    :return:
    """
    xp = get_array_module(x)
    with cuda.get_device_from_array(x.data):
        identity = xp.identity(number_of_types, xp.float32)
    return F.embed_id(x, identity)
