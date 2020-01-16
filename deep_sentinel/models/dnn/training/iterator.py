from typing import TYPE_CHECKING

from chainer.dataset import Iterator

if TYPE_CHECKING:
    from typing import Union, List
    from chainer.datasets import DictDataset, SubDataset


class ParallelSequentialIterator(Iterator):

    def __init__(self, dataset: 'Union[DictDataset, SubDataset]', batch_size: int, repeat: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        self.epoch = 0
        self.dataset_length = len(dataset)
        self.is_new_epoch = False
        self.offsets = [i * self.dataset_length // self.batch_size for i in range(self.batch_size)]
        self.iteration = 0
        self._previous_epoch_detail = -1

    def __next__(self):
        """
        Return minibatch sequence. Each element of minibatch indicates a different position
        in the original dataset sequence.
        :return:
        """
        if not self.repeat and self.iteration * self.batch_size >= self.dataset_length:
            raise StopIteration
        self._previous_epoch_detail = self.epoch_detail

        current_data = [self.dataset[self.get_position(o)] for o in self.offsets]

        self.iteration += 1
        current_epoch = self.iteration * self.batch_size // self.dataset_length
        self.is_new_epoch = self.epoch <= current_epoch
        if self.is_new_epoch:
            self.epoch = current_epoch
        return current_data

    @property
    def epoch_detail(self):
        """Floating point of epoch"""
        return self.iteration * self.batch_size / self.dataset_length

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def get_position(self, offset):
        """Obtain current position of dataset."""
        return (offset + self.iteration) % self.dataset_length

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        self._previous_epoch_detail = serializer('previous_epoch_detail', self._previous_epoch_detail)

    def reset(self):
        self.epoch = 0
        self.is_new_epoch = False
        self.iteration = 0
        self._previous_epoch_detail = -1


class MultipleSequenceIterator(Iterator):
    """Iterator for multiple DictDataset"""

    def __init__(self, datasets: 'List[Union[DictDataset, SubDataset]]', batch_ratio: int = 1, repeat: bool = True):
        if not isinstance(batch_ratio, int) and batch_ratio >= 1:
            raise TypeError("'batch_ratio' must be a positive integer. But actual '{}'".format(batch_ratio))
        if len(datasets) == 0:
            raise ValueError("'datasets' must have one or more elements at least.")
        # All dataset instance have the same length
        length = None
        for i in range(len(datasets)):
            if length is None:
                length = len(datasets[i])
            elif length != len(datasets[i]):
                raise ValueError("dataset length conflict at {} th attr.".format(i))
        self.datasets = [ParallelSequentialIterator(d, batch_ratio, repeat) for d in datasets]

    def __next__(self):
        """
        Return minibatch sequence. Each element of minibatch indicates a different position
        in the original dataset sequence.
        :return:
        """
        # Flatten all list of dataset
        current_data = sum([next(d) for d in self.datasets], [])
        return current_data

    @property
    def iteration(self):
        return self.datasets[0].iteration

    @property
    def epoch(self):
        return self.datasets[0].epoch

    @property
    def is_new_epoch(self):
        return self.datasets[0].is_new_epoch

    @property
    def epoch_detail(self):
        return self.datasets[0].epoch_detail

    @property
    def previous_epoch_detail(self):
        return self.datasets[0].previous_epoch_detail

    def reset(self):
        for d in self.datasets:
            d.reset()

    def serialize(self, serializer):
        for d in self.datasets:
            d.serialize(serializer)
