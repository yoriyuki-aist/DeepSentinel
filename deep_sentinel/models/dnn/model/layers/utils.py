from typing import TYPE_CHECKING

import numpy as np
from chainer import functions as F

if TYPE_CHECKING:
    from chainer import Variable


def reshape_batch(batch: 'Variable', n_batch_axes: int) -> 'Variable':
    if batch.ndim == n_batch_axes:
        return batch
    shape = (np.prod(batch.shape[:n_batch_axes]), -1)
    return F.reshape(batch, shape)
