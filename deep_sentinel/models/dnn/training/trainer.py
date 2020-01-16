import logging
from typing import TYPE_CHECKING

from chainer.optimizers import Adam
from chainer.training import Trainer as ChainerTrainer, extensions

from .evaluator import Validator
from .iterator import ParallelSequentialIterator, MultipleSequenceIterator
from .updater import Updater

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Type, List, Tuple, Optional, Union, Callable
    from chainer.datasets import DictDataset, SubDataset
    from chainer.optimizer import Optimizer

    ds_obj = Union[DictDataset, SubDataset]
    ds_objs = Union[ds_obj, List[ds_obj]]
    Iterators = Union[ParallelSequentialIterator, MultipleSequenceIterator]

logger = logging.getLogger(__name__)


def create_iterator(dataset: 'ds_objs',
                    batch_size: int = None, train: bool = True) -> 'Iterators':
    if isinstance(dataset, list):
        return MultipleSequenceIterator(dataset, batch_size, repeat=train)
    return ParallelSequentialIterator(dataset, batch_size, repeat=train)


class Trainer(object):

    def __init__(self, model, train_data: 'ds_objs', val_data: 'ds_objs',
                 device: int, batch_size: int, bprop_length: int,
                 output_dir: 'Path', optimizer_class: 'Type[Optimizer]' = None):
        """
        Initialize
        :param model: Model object which has `__call__()` to get loss.
        :param train_data: List or an instance of DictDataset for training
        :param val_data: List or an instance of DictDataset for validation
        :param device: GPU ID. If you specify -1, use CPU to train.
        :param batch_size: Mini batch size
        :param bprop_length:    Back propagation limit length
        :param output_dir: Where to output
        :param optimizer_class: Optimizer class to use. Default is `chainer.optimizers.Adam`.
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.bprop_length = bprop_length
        self.output_dir = output_dir
        self.optimizer_class = optimizer_class
        if self.optimizer_class is None:
            self.optimizer_class = Adam
        self.optimizer_hooks = list()
        self.trainer_extensions = list()
        self.disable_extensions = False

        # Hold the data to fit
        self.train_data = train_data
        self.val_data = val_data

    def set_optimizer_hook(self, hook) -> None:
        self.optimizer_hooks.append(hook)

    def _snapshots_list(self) -> 'Optional[List[Tuple[int, Path]]]':
        snapshots = list(self.output_dir.glob("snapshot-epoch-*"))
        if len(snapshots) == 0:
            return None
        with_epochs = [(int(s.name.split('-')[-1]), s) for s in snapshots]
        with_epochs.sort(key=lambda x: x[0])
        return with_epochs

    def get_best_snapshot(self) -> 'Optional[Path], Optional[float]':
        snapshots = self._snapshots_list()
        if snapshots is None:
            return None, None
        log_reporter = None
        for extension in self.trainer_extensions:
            if extension['name'] == 'LogReport':
                log_reporter = extension['extension']
        if log_reporter is None:
            raise AttributeError("Trainer has no LogReporter")
        observed_log = log_reporter.log
        observed_log.sort(key=lambda x: x['val/main/loss'])
        best_epoch = observed_log[0]
        for snapshot in snapshots:
            if snapshot[0] == best_epoch['epoch']:
                return snapshot[1], best_epoch['val/main/loss']
        return None, None

    def disable_builtin_extensions(self):
        self.disable_extensions = True

    def _setup_extensions(self, log_trigger, progress_interval):
        self.add_trainer_extension("LogReport", extensions.LogReport(trigger=log_trigger))
        self.add_trainer_extension("snapshot", extensions.snapshot(filename='snapshot-epoch-{.updater.epoch}'))
        if self.disable_extensions:
            return
        self.add_trainer_extension("PrintReport",
                                   extensions.PrintReport(
                                       ['epoch', 'iteration', 'main/loss', 'val/main/loss', 'elapsed_time']
                                   ), {'trigger': log_trigger})
        self.add_trainer_extension("PlotReport",
                                   extensions.PlotReport(['main/loss', 'val/main/loss'],
                                                         x_key=log_trigger[-1], file_name='loss.png'))
        self.add_trainer_extension("ProgressBar", extensions.ProgressBar(update_interval=progress_interval))

    def add_trainer_extension(self, name, extension, kwargs: 'Optional[dict]' = None):
        """
        Add additional extensions to Trainer.
        :param name: Extension name (just use to logging)
        :param extension: Extension instance
        :param kwargs: Additional arguments for `Trainer.extend()`
        :return: None
        """
        if kwargs is None:
            kwargs = dict()
        self.trainer_extensions.append(
            {
                "name": name,
                "extension": extension,
                "kwargs": kwargs
            }
        )

    def run(self, until: 'Union[Callable[[Trainer], bool], Tuple[int, str]]',
            log_trigger: 'Tuple[int, str]' = (1, 'epoch'),
            progress_interval: int = 100, optimizer_args: 'dict' = None):
        """
        Start to learn
        :param until: How long will you continue learning. ex. (10, 'epoch') or (1000, 'iteration')
        :param log_trigger: Frequency of logging. ex. (1, 'epoch') or (100, 'iteration')
        :param progress_interval: Show progress per this interval (iteration).
        :param optimizer_args: Arguments for optimizer class constructor.
        :return: Trained model
        """
        train_iter = create_iterator(self.train_data, self.batch_size)
        valid_iter = create_iterator(self.val_data, 1, train=False)
        if optimizer_args is None:
            optimizer_args = {}
        optimizer = self.optimizer_class(**optimizer_args).setup(self.model)

        for hook in self.optimizer_hooks:
            optimizer.add_hook(hook)

        updater = Updater(self.bprop_length, train_iter, optimizer, device=self.device)
        # updater = StandardUpdater(train_iter, optimizer, device=self.device)
        trainer = ChainerTrainer(updater, until, out=str(self.output_dir))

        # Setup extensions
        self._setup_extensions(log_trigger, progress_interval)
        eval_model = self.model.copy()
        self.add_trainer_extension("Evaluator",
                                   Validator(iterator={'main': valid_iter},
                                             target=eval_model, device=self.device),
                                   {'name': 'val'})
        logger.debug(f"Logging triggered per {log_trigger[0]} {log_trigger[1]}")
        for extension in self.trainer_extensions:
            logger.debug(f"Enable '{extension['name']}' extension")
            trainer.extend(extension['extension'], **extension['kwargs'])

        trainer.run()
        del trainer
        return self.model
