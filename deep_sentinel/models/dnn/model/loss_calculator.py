from typing import TYPE_CHECKING

from chainer import functions as F, links as L
from chainer import reporter

if TYPE_CHECKING:
    from typing import Optional, Tuple
    from chainer import Variable


def calculate_continuous_value_loss(predicted: 'Variable', actual: 'Variable') -> 'Variable':
    """
    Calculate loss according to predicted results
    :param predicted: Predicted value.
    :param actual: Actual results
    :return: loss
    """
    if predicted.shape[-1] > 2:
        return calculate_gmm_nll(predicted, actual)
    mean, scale = F.separate(predicted, axis=-1)
    loss = F.gaussian_nll(actual, mean, scale, reduce='no')
    # Return the mean of loss for each time record
    return loss


def calculate_discrete_value_loss(predicted: 'Variable', actual: 'Variable') -> 'Variable':
    """
    Calculate loss according to predicted results
    :param predicted: Predicted value.
    :param actual: Actual results
    :return: loss
    """
    batch_size = actual.shape[0]
    time_length = actual.shape[1]
    columns = actual.shape[2]
    actual = actual.reshape(batch_size * time_length, columns)
    predicted = F.transpose(predicted, (0, 1, 3, 2)) \
        .reshape(batch_size * time_length, -1, columns)
    loss = F.softmax_cross_entropy(predicted, actual, reduce='no')
    return loss.reshape(batch_size, time_length, -1)


def calculate_gmm_nll(predicted: 'Variable', actual: 'Variable') -> 'Variable':
    """
    Calculate negative log likelihood of the given data in GMM.
    :param predicted: Predicted value.
    :param actual: Actual results
    :return:
    """
    mean, scale, weights = F.split_axis(predicted, 3, -1)
    gmm_class_count = mean.shape[-1]
    actual = F.broadcast_to(F.expand_dims(actual, -1), (*actual.shape, gmm_class_count))
    # Convert weights to the probabilities
    mix_ratio = F.log(F.softmax(weights, axis=-1))
    loss = F.gaussian_nll(actual, mean, scale, reduce='no') + mix_ratio
    # Sum along the axis of the number of class of GMM
    loss = F.logsumexp(loss, axis=-1)
    return loss


class LossCalculator(L.Classifier):
    """Wrapper for loss calculation and log reports"""

    def __init__(self, *args, **kwargs):
        super(LossCalculator, self).__init__(*args, **kwargs)
        self.compute_accuracy = False
        self.is_training = True

    def set_as_predict(self):
        self.is_training = False

    def __call__(self, current_continuous: 'Variable', next_continuous: 'Variable',
                 current_discrete: 'Optional[Variable]' = None, next_discrete: 'Optional[Variable]' = None):
        self.loss = None
        if current_discrete is not None and next_discrete is not None:
            predicted_values, predicted_discretes = self.predictor(
                current_continuous, next_continuous, current_discrete, next_discrete
            )
            pos_loss = calculate_discrete_value_loss(predicted_discretes, next_discrete)
            cont_loss = calculate_continuous_value_loss(predicted_values, next_continuous)
            self.loss = F.concat([pos_loss, cont_loss], axis=-1)
        else:
            predicted_values, _ = self.predictor(current_continuous, next_continuous)
            self.loss = calculate_continuous_value_loss(predicted_values, next_continuous)
        if self.is_training:
            # Calculate mean loss for each mini batch and time length
            self.loss = F.mean(self.loss)
            reporter.report({'loss': self.loss}, self)
        return self.loss

    def initialize_with(self, initial_continuous: 'Variable', initial_discrete: 'Variable' = None):
        if initial_discrete is None:
            return self.predictor.initialize_with(initial_continuous)
        else:
            return self.predictor.initialize_with(initial_continuous, initial_discrete)

    def sample(self, steps: int = 1) -> 'Tuple[Variable, Variable]':
        return self.predictor.sample(steps)

    def reset_state(self):
        self.predictor.reset_state()
