import argparse
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, auc

from deep_sentinel import utils
from deep_sentinel.models import dnn

try:
    from .dataset import SWaTData
except:
    import sys
    path = str(Path(__file__).parent)
    if path not in sys.path:
        sys.path.append(path)
    from dataset import SWaTData

if TYPE_CHECKING:
    from typing import Optional, Union

    NullableDF = Optional[pd.DataFrame]

    import numpy as np

score_column_name = 'scores'
precision_label = 'Precision'
recall_label = 'Recall'
f_measure_label = 'F measure'


class AttackReport(object):
    date_format = '%Y/%m/%d %H:%M:%S'

    def _parse_date(self, date_str: 'Union[datetime, str]') -> 'datetime':
        if isinstance(date_str, datetime):
            return date_str
        return datetime.strptime(date_str, self.date_format)

    def __init__(self, number: int, from_date: 'Union[datetime, str]', to_date: 'Union[datetime, str]',
                 start_state: str, attack_point: str, attack: str, actual_change: str,
                 expected: str, unexpected: str):
        self.number = number
        self.from_date = self._parse_date(from_date)
        self.to_date = self._parse_date(to_date)
        self.start_state = start_state
        self.attack_point = attack_point
        self.attack = attack
        self.actual_change = actual_change
        self.expected = expected
        self.unexpected = unexpected

    def report(self, score_with_labels: 'pd.DataFrame', threshold) -> 'pd.DataFrame':
        target_scores = score_with_labels[self.from_date:self.to_date]
        target_scores["Normal"] = target_scores.loc[:, score_column_name] >= threshold * 1
        truth = target_scores['Attack'].values
        pred = target_scores['Normal'].values
        return pd.DataFrame({
            # zero_division=1 means that suppress warning when the calculation causes zero-division
            precision_label: [precision_score(truth, pred, zero_division=1)],
            recall_label: [recall_score(truth, pred, zero_division=1)],
            f_measure_label: [f1_score(truth, pred, zero_division=1)],
            'Start Time': [self.from_date],
            'End Time': [self.to_date],
            'Attack Point': [self.attack_point],
            'Start State': [self.start_state],
            'Attack': [self.attack],
            'Actual Change': [self.actual_change],
            'Expected Impact or Attacker Intent': [self.expected],
            'Unexpected Outcome': [self.unexpected]
        }, index=[self.number])


class SWaTClassificationMetrics(object):
    precision_label = 'Precision'
    recall_label = 'Recall'
    f_measure_label = 'F measure'
    auc_label = 'AUC'
    roc_label = 'ROC'
    roc_true_label = 'True Positive Rate'
    roc_false_label = 'False Positive Rate'
    normal_label = 'Normal'
    attack_label = 'Attack'

    def __init__(self, predicted_results: 'pd.DataFrame', true_values: 'pd.DataFrame'):
        """
        Initialize. Predicted results must be `pandas.DataFrame` and its shape is (T, 1),
        where `T` is a index value of time series.
        :param predicted_results:
        :param true_values:
        """
        self.predicted_results = predicted_results
        self.true_values = true_values
        self.results_with_labels = pd.concat([predicted_results, true_values], axis=1, join='inner')

        # Sort by scores
        self.results_with_labels = self.results_with_labels.sort_values(
            by=score_column_name, ascending=False
        )

        # List of attacks
        self.attack_list = Path(__file__).parent / 'attack_list.csv'

    @property
    @lru_cache()
    def f_max_id(self):
        return self.f_measure.idxmax()

    @property
    @lru_cache()
    def correct_detection(self) -> 'pd.Series':
        return self.results_with_labels[self.attack_label].cumsum()

    @property
    @lru_cache()
    def false_detection(self) -> 'pd.Series':
        return self.results_with_labels[self.normal_label].cumsum()

    @property
    @lru_cache()
    def precision(self) -> 'pd.Series':
        _precision = self.correct_detection / (self.correct_detection + self.false_detection)
        _precision.name = self.precision_label
        return _precision

    @property
    @lru_cache()
    def recall(self) -> 'pd.Series':
        _recall = self.correct_detection / self.results_with_labels[self.attack_label].sum()
        _recall.name = self.recall_label
        return _recall

    @property
    @lru_cache()
    def false_positive(self) -> 'pd.Series':
        return self.false_detection / self.results_with_labels[self.normal_label].sum()

    @property
    @lru_cache()
    def f_measure(self) -> 'pd.Series':
        _f_measure = 2 * self.precision * self.recall / (self.precision + self.recall)  # type: pd.Series
        _f_measure.name = self.f_measure_label
        return _f_measure.fillna(0)

    @property
    @lru_cache()
    def roc(self) -> 'pd.DataFrame':
        df = pd.DataFrame(index=self.predicted_results.index)
        df[self.roc_false_label] = self.false_positive
        df[self.roc_true_label] = self.recall
        return df

    @property
    @lru_cache()
    def auc(self) -> 'float':
        return auc(self.false_positive.values, self.recall.values)

    @property
    @lru_cache()
    def summary(self):
        f_max_id = self.f_max_id
        return pd.DataFrame.from_dict({
            self.precision_label: [self.precision[f_max_id]],
            self.recall_label: [self.recall[f_max_id]],
            self.f_measure_label: [self.f_measure[f_max_id]],
            self.auc_label: [self.auc]
        })

    @property
    @lru_cache()
    def detail(self):
        threshold = self.results_with_labels.loc[self.f_max_id][score_column_name]
        attacks = pd.read_csv(str(self.attack_list), header=[0])
        df_set = []
        for row in attacks.itertuples():
            # row is a named-tuple and first element is an Index obj
            attack = AttackReport(*[r for r in row[1:]])
            df_set.append(attack.report(self.results_with_labels.sort_index(), threshold))
        return pd.concat(df_set)


def create_score_dataframe(scores: 'np.ndarray', index) -> 'pd.DataFrame':
    return pd.DataFrame({score_column_name: scores}, index=index[1:])


def get_parser():
    parser = argparse.ArgumentParser(description="Sample implementation for anomaly detection of SWaT system"
                                                 " using DeepSentinel")
    parser.add_argument("--attack",
                        type=Path,
                        required=True,
                        help="Path to attack data")
    parser.add_argument("-o",
                        "--output-dir",
                        type=Path,
                        required=True,
                        help="Path to output dir")
    parser.add_argument("-d",
                        "--device",
                        type=str,
                        default="-1",
                        help="Device ID to use (negative value indicate CPU)")
    parser.add_argument("-m",
                        "--model",
                        type=Path,
                        required=True,
                        help="Path to model file")
    parser.add_argument("--steps",
                        type=int,
                        default=100,
                        help="Number of steps to sample")
    parser.add_argument("--trials",
                        type=int,
                        default=1000,
                        help="Number of trials to sample")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    out_dir = utils.mkdir(args.output_dir)

    attack_file = utils.to_absolute(args.attack)
    if not attack_file.exists():
        print("{} does not exist.".format(attack_file))
        exit(1)

    trained_model = utils.to_absolute(args.model)
    if not trained_model.exists():
        print("{} does not exist.".format(trained_model))
        exit(1)

    # Create model instance
    # These default values are override when loading existing model.
    dnn_model = dnn.DNN(
        batch_size=64,
        device=args.device,
        n_units=64,
        lstm_stack=1,
        dropout_ratio=0.5,
        activation='sigmoid',
        bprop_length=100,
        max_epoch=20,
        output_dir=out_dir,
        gmm_classes=1
    )
    print("----- Load trained model -----")
    print("Use: {}".format(trained_model))
    # This step will require the saved weights (`dnn-model` file) and its metadata (`dnn-model-meta` file)
    # If Chainer's snapshot file is passed, this still require the metadata file (`dnn-model-meta` file)
    dnn_model.load(trained_model)
    print("----- End -----")

    print("----- Model Params -----")
    print("N units: {}".format(dnn_model.n_units))
    print("LSTM Stack: {}".format(dnn_model.lstm_stack))
    print("Back propagation length: {}".format(dnn_model.bprop_length))
    print("Activation Function: {}".format(dnn_model.activation))
    print("GMM Classes: {}".format(dnn_model.gmm_classes))

    print("Try to read {} ......".format(attack_file))
    attack_data = SWaTData(attack_file)
    attack_df = attack_data.read()

    print("----- Start to scoring -----")
    print("File: {}".format(attack_file))
    print("Length: {}".format(len(attack_df)))
    label_columns = ['Attack', 'Normal']
    scores = dnn_model.score_samples(attack_df.drop(label_columns, axis=1))
    print("----- End -----")

    # Convet np.ndarray to pandas.DataFrame
    score_df = create_score_dataframe(scores, attack_df.index.values)
    # Calculate some metrics (Precision, Recall, and so on)
    metrics = SWaTClassificationMetrics(score_df, attack_df[label_columns])
    summary = metrics.summary
    print("----- Metrics Summary -----")
    for k, v in summary.to_dict().items():
        print("{k:>10}: {v[0]}".format(k=k, v=v))

    # Summary contains Precision/Recall/F1 measure/AUC values
    summary_file = out_dir / 'metrics_summary.csv'
    print("Save as: {}".format(summary_file))
    summary.to_csv(str(summary_file), quoting=2)

    # Detail includes Precision/Recall/F1 measure/AUC values for each cause of anomaly.
    detail_file = out_dir / 'metrics_detail.csv'
    detail = metrics.detail
    print("Save details as: {}".format(detail_file))
    detail.to_csv(str(detail_file))
    print("----- End -----")


if __name__ == '__main__':
    main()
