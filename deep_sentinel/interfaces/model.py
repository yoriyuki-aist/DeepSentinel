import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from deep_sentinel import dataset

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional, Union, Tuple, List
    import numpy as np
    import pandas as pd


logger = logging.getLogger(__name__)


class Model(metaclass=ABCMeta):

    def __init__(self):
        self.mean = None  # type: Optional[pd.Series]
        self.std = None  # type: Optional[pd.Series]

    def _register_train_data(self, x: 'pd.DataFrame') -> None:
        """Register the mean and std of train data"""
        self.mean = dataset.get_mean(x)
        self.std = dataset.get_std(x)

    def _normalize(self, x: 'pd.DataFrame') -> 'pd.DataFrame':
        assert self.mean is not None, "Calculate mean before normalization."
        assert self.std is not None, "Calculate std before normalization."
        return dataset.normalize(x, self.mean, self.std)

    def set_metadata(self, metadata: 'dict'):
        for key, val in metadata.items():
            current = getattr(self, key)
            if current is not None and current != val:
                logger.warning(f"Override {key}: {current} -> {val}")
            setattr(self, key, val)

    @abstractmethod
    def fit(self, x: 'pd.DataFrame') -> 'Tuple[Model, Union[float, np.ndarray]]':
        """
        Train the model with given dataset.
        :param x:   `pandas.DataFrame` object which has N rows * K columns.
                    N means time length, and K means number of features.
                    If you want to use categorical data, you should convert
                    the data which you want to use as category to `category`
                    from `int` or `float`.
        :return:    Trained model
        """
        raise NotImplementedError

    @abstractmethod
    def fit_all(self, x: 'List[pd.DataFrame]', **kwargs) -> 'Tuple[Model, Union[float, np.ndarray]]':
        """
        Train the model with given list of a dataset.
        :param x:       List of `pandas.DataFrame` object which has N rows * K columns.
                        N means time length, and K means number of features.
                        If you want to use categorical data, you should convert
                        the data which you want to use as category to `category`
                        from `int` or `float`.
                        All element of given list must have same size and shape.
        :param kwargs:  Some options to train. Please read the docs of implementation.
        :return:        Trained model and validation loss
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: 'Path') -> 'Path':
        """
        Save trained model as file.
        :param path: Path to directory to save the model.
        :return: Path of saved model
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: 'Path') -> 'Model':
        """
        Load trained model from file.
        :param path:    Path to directory which includes the trained model dump file.
        :return:        Trained model
        """
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, x: 'pd.DataFrame') -> 'np.ndarray':
        """
        Calculate score from given dataset `x` for each time record.
        :param x:   `pandas.DataFrame` object which has N rows * K columns.
                    N means time length, and K means number of features.
                    If you want to use categorical data, you should convert
                    the data which you want to use as category to `category`
                    from `int` or `float`.
        :return:    Calculated scores. The shape will be `(N, 1)`. `N` means time length.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, steps: int, trials: int = 1) -> 'np.ndarray':
        """
        Execute multi step prediction. Use `initialize_with` to initialize the state.
        If you initialize this model with `N` records, `predict` returns the values started from `N+1`.
        :param steps:   Number of steps to predict.
        :param trials:  Number of trials. If you specify the params as
                        `steps=N` and `trials=M`, `N` step prediction will
                        be executed with `M` times. The model to use is
                        deep-copied for each trial.
        :return:        Predicted values. The shape will be `(M, N, K)`.
                        `M` means trial number and `N` means time length,
                        `K` means the number of features.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_with(self, x: 'pd.DataFrame') -> 'Model':
        """
        Initialize the state in the model with given dataset `x`.
        This method should be called before prediction.
        :param x:   `pandas.DataFrame` object which has N rows * K columns.
                    N means time length, and K means number of features.
                    If you want to use categorical data, you should convert
                    the data which you want to use as category to `category`
                    from `int` or `float`.
        :return:    Initialized model
        """
        raise NotImplementedError

    @abstractmethod
    def clean_artifacts(self, output_dir: 'Path') -> 'None':
        """
        This method will be called after one of the trial of optimization.
        Erase unused intermediate files if need.
        :param output_dir: Path to output directory.
        :return: None
        """
        raise NotImplementedError
