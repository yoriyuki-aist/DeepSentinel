import logging
import pickle
import shutil
from pkg_resources import parse_version
from pathlib import Path
from typing import TYPE_CHECKING

import chainer
import numpy as np
import pandas as pd
from chainer import serializers, Variable, functions as F
from chainer.backends import cuda
from chainer.dataset import concat_examples
from chainer.training.triggers import EarlyStoppingTrigger

from deep_sentinel import dataset
from deep_sentinel.interfaces import Model
from deep_sentinel.utils import parallel, to_path
from .dataset import create_dataset, split_dataset, extract_from
from .model import DeepSentinel, DeepSentinelWithoutDiscrete, LossCalculator
from .training import Trainer, create_iterator

if TYPE_CHECKING:
    from typing import Optional, Union, Tuple, List
    from chainer.datasets import DictDataset


logger = logging.getLogger(__name__)


def _cut_array(arr: 'np.ndarray', length: int) -> 'np.ndarray':
    # The shape of array is `(trials, time length, data dimension)`
    if arr.shape[0] > length:
        return arr[:length]
    return arr


class DNN(Model):
    """Wrapper module of DNN model. Organize training and prediction tasks."""

    save_model_name = "dnn-model"
    meta_data_name = "dnn-model-meta"

    # Number of batch for sampling
    TRIAL_BATCH = 1000

    def __init__(self, batch_size: int, device: int, n_units: int, lstm_stack: int, dropout_ratio: float,
                 activation: str, bprop_length: int, max_epoch: int, output_dir: 'Union[str, Path]',
                 gmm_classes: int):
        super(DNN, self).__init__()
        logger.debug("Model params: " + str(
            {
                "batch_size": batch_size,
                "device": device,
                "n_units": n_units,
                "lstm_stack": lstm_stack,
                "dropout_ratio": dropout_ratio,
                "activation": activation,
                "bprop_length": bprop_length,
                "max_epoch": max_epoch,
                "gmm_classes": gmm_classes,
            }
        ))
        self.batch_size = batch_size
        self.device = device
        self.n_units = n_units
        self.lstm_stack = lstm_stack
        self.dropout_ratio = dropout_ratio
        self.activation = activation.lower()
        self.bprop_length = bprop_length
        self.max_epoch = max_epoch
        self.gmm_classes = gmm_classes
        self.output_dir = to_path(output_dir)

        # Actual model instance
        self._model = None  # type: Optional[LossCalculator]

        # For validation
        self._continuous_column_count = None  # type: Optional[int]
        self._discrete_column_count = None  # type: Optional[int]
        self._discrete_state_kinds = None  # type: Optional[int]

        # For disable outputs
        self._disable_extensions = False

        # Additional extensions for training
        self._additional_exts = {}

    def _create_dataset(self, x: 'pd.DataFrame') -> 'DictDataset':
        continuous, discrete = dataset.split_category_data(x)
        # Record the metrics of training data
        self._continuous_column_count = len(continuous.columns)
        self._discrete_column_count = len(discrete.columns)
        if self._discrete_column_count > 0:
            self._discrete_state_kinds = int(discrete.nunique().max())
        dict_dataset = create_dataset(self._normalize(continuous), discrete, self.bprop_length)
        return dict_dataset

    def set_trainer_extension(self, name: str, extension: 'chainer.training.Extension'):
        self._additional_exts[name] = extension

    def _fit(self, train_data, val_data, batch_size: 'Optional[int]' = None) -> 'Optional[float, np.ndarray]':
        self.setup_model()

        trigger = EarlyStoppingTrigger(
            monitor='val/main/loss', patients=10, mode='min',
            max_trigger=(self.max_epoch, 'epoch')
        )

        trainer = self.get_trainer(train_data, val_data, str(self.output_dir))

        if self._disable_extensions:
            trainer.disable_builtin_extensions()

        for name, ext in self._additional_exts.items():
            trainer.add_trainer_extension(name, ext)

        # Override batch size
        if batch_size:
            trainer.batch_size = batch_size

        trainer.run(until=trigger)
        best_snapshot, validation_loss = trainer.get_best_snapshot()

        shutil.copy(str(best_snapshot), str(best_snapshot.parent / self.save_model_name))
        self.load(best_snapshot)
        return validation_loss

    def fit(self, x: 'pd.DataFrame') -> 'Tuple[Model, Optional[float, np.ndarray]]':
        self._register_train_data(x)
        dict_dataset = self._create_dataset(x)
        train_data, val_data = split_dataset(dict_dataset)
        validation_loss = self._fit(train_data, val_data)
        return self, validation_loss

    def fit_all(self, x: 'List[pd.DataFrame]', batch_ratio: int = 1) -> 'Tuple[Model, Optional[float, np.ndarray]]':
        if not isinstance(batch_ratio, int) or batch_ratio < 1:
            raise ValueError("'batch_ratio' must be a positive integer.")
        self._register_train_data(pd.concat(x, axis=0))
        dict_datasets = [self._create_dataset(_x) for _x in x]

        # Find maximal batch ratio if the given one is too large.
        data_length = len(dict_datasets[0])
        orig_batch_ratio = batch_ratio
        while True:
            if data_length > batch_ratio:
                break
            batch_ratio -= 1
        if batch_ratio != orig_batch_ratio:
            logger.warning('Specified batch_ratio is too large. '
                           'Use {} instead of {}'.format(batch_ratio, orig_batch_ratio))
            batch_ratio = 1

        # Check the actual minibatch size
        batch_size = len(dict_datasets) * batch_ratio
        if self.batch_size != batch_size:
            logger.warning("Actual minibatch size becomes {} not {}".format(batch_ratio, self.batch_size))

        separated = [split_dataset(d) for d in dict_datasets]
        train_data = [s[0] for s in separated]
        val_data = [s[1] for s in separated]
        validation_loss = self._fit(train_data, val_data, batch_ratio)
        return self, validation_loss

    def get_metadata(self) -> dict:
        return {
            'n_units': self.n_units,
            'lstm_stack': self.lstm_stack,
            'activation': self.activation,
            'bprop_length': self.bprop_length,
            'gmm_classes': self.gmm_classes,
            '_continuous_column_count': self._continuous_column_count,
            '_discrete_column_count': self._discrete_column_count,
            '_discrete_state_kinds': self._discrete_state_kinds,
            'mean': self.mean,
            'std': self.std
        }

    def save(self, path: 'Path') -> 'Path':
        assert self._model is not None, "Model is not trained yet"
        path = to_path(path)
        if path.is_dir():
            best_model_dump = path / self.save_model_name
            metadata_dump = path / self.meta_data_name
            parent_dir = path
        else:
            best_model_dump = path
            metadata_dump = path / self.meta_data_name
            parent_dir = path.parent
        # Save model params as `npz` format
        serializers.save_npz(str(best_model_dump), self._model)
        with open(str(metadata_dump), 'wb') as fp:
            pickle.dump(self.get_metadata(), fp)

        if self.output_dir.samefile(path):
            return best_model_dump
        # Copy all files into output dir from temporary dir
        for tmp_file in self.output_dir.glob("*"):
            shutil.copy2(str(tmp_file), str(parent_dir / tmp_file.name))
        return best_model_dump

    def load(self, path: 'Path') -> 'Model':
        path = to_path(path)
        if path.is_dir():
            metadata = path / self.meta_data_name
            path = path / self.save_model_name
        else:
            metadata = path.parent / self.meta_data_name
        if path.name == self.save_model_name:
            load_path = ''
        else:
            # When snapshot file is specified
            # Snapshot file is a snapshot of Trainer.
            # So the path is different from model only dump.
            load_path = 'updater/model:main/'
        if self._model is None:
            with open(str(metadata), 'rb') as fp:
                meta = pickle.load(fp)
            self.set_metadata(meta)
            self._model = self.get_model()
        serializers.load_npz(str(path), self._model, path=load_path)
        return self

    def score_samples(self, x: 'pd.DataFrame') -> 'np.ndarray':
        tmp = self.bprop_length
        self.bprop_length = tmp * 10
        dict_dataset = self._create_dataset(x)
        self.setup_model()
        model = self._model.copy(mode='copy')
        if self.device >= 0:
            self._model_to_gpu(model)
        model.set_as_predict()
        iterator = create_iterator(dict_dataset, 1, train=False)
        results = []
        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                while True:
                    try:
                        batch = iterator.next()
                    except StopIteration:
                        break
                    values = concat_examples(batch, self.device)
                    vals = model(*extract_from(values)).data
                    if self.device >= 0:
                        vals = cuda.to_cpu(vals)
                    vals = np.sum(vals, axis=-1, keepdims=True)
                    results.append(np.squeeze(vals))
        # The shape is `(time length,)`
        results = np.concatenate(results, axis=0)
        self.bprop_length = tmp
        return results

    def _sample(self, steps: int) -> 'Tuple[np.ndarray, Optional[np.ndarray]]':
        model = self._model.copy(mode='copy')
        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                vals = model.sample(steps)
                # Shape of sampled items is `(batch, time length, features)`
                vals = [v.data if v is not None else None for v in vals]
                if chainer.backend.get_array_module(vals[0]) != np:
                    vals = [cuda.to_cpu(v) if v is not None else None for v in vals]
        return vals

    def sample(self, steps: int, trials: int = 1) -> 'np.ndarray':
        # Batch prediction is supported to improve performance
        batch_trials = trials // self.TRIAL_BATCH
        # If trial counts is not divisible by batch size,
        # increment batch trial to satisfy the requirement of trial count.
        if trials % self.TRIAL_BATCH != 0:
            batch_trials += 1
        results = parallel(self._sample, [steps for _ in range(batch_trials)], 1)
        continuous = [r[0] for r in results]
        discrete = [r[1] for r in results]
        continuous = _cut_array(np.vstack(continuous), trials)
        # Restore normalized values
        predicted_values = continuous * self.std.values + self.mean.values
        # If the model supports prediction of discrete values
        if discrete[0] is not None:
            discrete = _cut_array(np.vstack(discrete), trials)
            predicted_values = np.hstack([predicted_values, discrete])
        return predicted_values

    def initialize_with(self, x: 'pd.DataFrame') -> 'Model':
        self._model.reset_state()
        continuous, discrete = dataset.split_category_data(x)
        continuous = self._normalize(continuous)
        continuous = Variable(continuous.astype('float32').values)
        discrete = Variable(discrete.astype('float32').values)
        if parse_version(chainer.__version__[0]) >= parse_version("6.0.0"):
            device = chainer.get_device(self.device)
            continuous.to_device(device)
            discrete.to_device(device)
        else:
            if self.device >= 0:
                continuous.to_gpu(self.device)
                discrete.to_gpu(self.device)
        continuous = F.expand_dims(continuous, 0)
        discrete = F.expand_dims(discrete, 0)
        continuous = F.repeat(continuous, self.TRIAL_BATCH, 0)
        discrete = F.repeat(discrete, self.TRIAL_BATCH, 0)
        self.setup_model()
        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                if self._discrete_column_count > 0:
                    self._model.initialize_with(continuous, discrete)
                else:
                    self._model.initialize_with(continuous)
        return self

    def setup_model(self):
        if self._model is None:
            self._model = self.get_model()
        # If chainer 6.0.0 or later is used, use chainer's Device feature.
        if parse_version(chainer.__version__[0]) >= parse_version("6.0.0"):
            device = chainer.get_device(self.device)
            self._model.to_device(device)
            device.use()
        else:
            if self.device >= 0:
                cuda.get_device_from_id(self.device).use()
                self._model.to_gpu(self.device)
        return self._model

    def get_model(self) -> 'LossCalculator':
        if self._discrete_column_count > 0:
            model = DeepSentinel(
                continuous_unit_count=self._continuous_column_count,
                discrete_unit_count=self._discrete_column_count,
                discrete_state_kinds=self._discrete_state_kinds,
                n_units=self.n_units,
                activation_func_type=self.activation,
                dropout_ratio=self.dropout_ratio,
                lstm_stack=self.lstm_stack
            )
        else:
            model = DeepSentinelWithoutDiscrete(
                continuous_unit_count=self._continuous_column_count,
                n_units=self.n_units,
                activation_func_type=self.activation,
                dropout_ratio=self.dropout_ratio,
                lstm_stack=self.lstm_stack,
                gmm_class_count=self.gmm_classes
            )
        logger.debug("Model type: {}".format(model.__class__))
        return LossCalculator(model)

    def get_trainer(self, train_data, val_data, output_dir: str) -> 'Trainer':
        return Trainer(self._model, train_data, val_data, self.device,
                       self.batch_size, self.bprop_length, Path(output_dir))

    def disable_extensions(self) -> None:
        self._disable_extensions = True

    def clean_artifacts(self, output_dir: 'Path') -> 'None':
        # Dump of best model is saved as `dnn-model`. Remove snapshots for the capacity of disks.
        for f in output_dir.glob("snapshot-*"):
            f.unlink()
