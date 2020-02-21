from typing import TYPE_CHECKING

from chainer import Variable

from .constants import (
    CURRENT_CONTINUOUS, CURRENT_DISCRETE, NEXT_CONTINUOUS, NEXT_DISCRETE
)

if TYPE_CHECKING:
    from typing import Optional, Dict, Tuple
    import numpy as np


def _to_variable(arr: 'np.ndarray') -> 'Optional[Variable]':
    if arr is None:
        return arr
    return Variable(arr)


def extract_from(dataset: 'Dict[str, np.ndarray]') \
        -> "Tuple[Variable, Variable, Optional[Variable], Optional[Variable]]":

    current_continuous = _to_variable(dataset.get(CURRENT_CONTINUOUS))
    next_continuous = _to_variable(dataset.get(NEXT_CONTINUOUS))
    current_discrete = _to_variable(dataset.get(CURRENT_DISCRETE))
    next_discrete = _to_variable(dataset.get(NEXT_DISCRETE))

    return (
        current_continuous, next_continuous, current_discrete, next_discrete
    )
