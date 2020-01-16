from typing import TYPE_CHECKING

from chainer import reporter as reporter_module, function
from chainer.training import extensions

from deep_sentinel.models.dnn.dataset import extract_from

if TYPE_CHECKING:
    import numpy as np
    from typing import Dict


class Validator(extensions.Evaluator):
    """Customized evaluator"""

    def evaluate(self):
        iterator = self.get_iterator('main')
        eval_func = self.get_target('main')

        # Reset LSTM state before evaluation
        eval_func.reset_state()

        if self.eval_hook:
            self.eval_hook(self)

        iterator.reset()

        summary = reporter_module.DictSummary()

        for batch in iterator:
            observation = {}
            with reporter_module.report_scope(observation):
                with function.no_backprop_mode():
                    values = self.converter(batch, self.device)  # type: Dict[str, np.ndarray]
                    eval_func(*extract_from(values))
            summary.add(observation)
        return summary.compute_mean()
