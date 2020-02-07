from typing import TYPE_CHECKING

import chainer
import chainerx
from chainer.training import StandardUpdater

from deep_sentinel.models.dnn.dataset import extract_from

if TYPE_CHECKING:
    from typing import Dict
    import numpy as np


class Updater(StandardUpdater):
    """
    Update parameters of the model for each iteration.
    Call `unchain_backward()` if the iteration exceeds its limit.
    """

    def __init__(self, bprop_length: int, *args, **kwargs):
        """
        Initialize updater
        :param continuous_iterator: Iterator for continuous values
        :param position_iterator: Iterator for position values
        :param bprop_length: Back propagation limit length.
        :param args: Args for StandardUpdater
        :param kwargs: Kwargs for StandardUpdater
        """
        super(Updater, self).__init__(*args, **kwargs)
        self.bprop_length = bprop_length

    def update_core(self):
        iterator = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Reset LSTM state when new epoch is start
        if iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)
            optimizer.target.reset_state()

        batch = iterator.next()
        values = self.converter(batch, self.device)  # type: Dict[str, np.ndarray]
        data_tuple = extract_from(values)

        loss = optimizer.target(*data_tuple)

        optimizer.target.cleargrads()
        loss.backward()
        # Call `unchain_backward()` due to the limitation of compute resources.
        if chainer.backend.get_array_module(loss) != chainerx:
            # ChainerX does not support
            loss.unchain_backward()
        optimizer.update()
