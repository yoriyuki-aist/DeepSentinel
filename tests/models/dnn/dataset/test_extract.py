import numpy as np
import pytest

import numpy as np
from chainer import Variable

from deep_sentinel.models.dnn import dataset
from deep_sentinel.models.dnn.dataset import constants


@pytest.fixture(scope='session')
def dataset_keys():
    return [
        constants.CURRENT_CONTINUOUS,
        constants.NEXT_CONTINUOUS,
        constants.CURRENT_DISCRETE,
        constants.NEXT_DISCRETE,
    ]


@pytest.mark.parametrize(
    "data", [
        (np.arange(5), np.arange(5), None, None),
        (np.arange(5), np.arange(5), np.arange(5), np.arange(5)),
    ]
)
def test_extract_from(data, dataset_keys):
    actual = dataset.extract_from(dict(zip(dataset_keys, data)))
    for expected, actual in zip(data, actual):
        if expected is not None:
            assert isinstance(actual, Variable)
        else:
            assert actual is None
