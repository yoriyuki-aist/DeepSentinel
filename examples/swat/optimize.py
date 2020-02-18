import argparse
from pathlib import Path

import optuna
import pandas as pd

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

activation_funcs = dnn.model.utils.ActivationFunc.choices()


def get_parser():
    parser = argparse.ArgumentParser(description="Sample implementation to optimize the model"
                                                 " with SWaT dataset")
    parser.add_argument("--normal",
                        type=Path,
                        required=True,
                        help="Path to normal data")
    parser.add_argument("-o",
                        "--output-dir",
                        type=Path,
                        required=True,
                        help="Path to output dir")
    opt_params = parser.add_argument_group("Optimization params")
    opt_params.add_argument("-t",
                            "--trials",
                            type=int,
                            default=300,
                            help="Number of trials to optimize")
    opt_params.add_argument("-d",
                            "--device",
                            type=str,
                            default="-1",
                            help="Device ID to use (negative value indicate CPU)")
    opt_params.add_argument("-r",
                            "--resume",
                            default=False,
                            action="store_true",
                            help="Resume from existing optimize.db")
    opt_params.add_argument("-p",
                            "--prune",
                            default=False,
                            action="store_true",
                            help="Enable pruning feature of Optuna")
    return parser


def train(trial: 'optuna.Trial',
          train_data: 'pd.DataFrame',
          out_dir: 'Path',
          device: 'str',
          pruning_enabled: 'bool') -> 'float':
    out_dir = out_dir / str(trial.number)
    # Create model instance with the params which is suggested by Optuna
    dnn_model = dnn.DNN(
        batch_size=trial.suggest_int('batch_size', 8, 128),
        output_dir=out_dir,
        max_epoch=50,
        device=device,
        n_units=trial.suggest_int('n_units', 16, 256),
        lstm_stack=trial.suggest_int('lstm_stack', 1, 10),
        dropout_ratio=trial.suggest_uniform('dropout_ratio', 0, 1),
        activation=trial.suggest_categorical('activation', activation_funcs),
        bprop_length=trial.suggest_int('bprop_length', 10, 200),
        gmm_classes=trial.suggest_int('gmm_classes', 1, 5)
    )
    # Disable some extensions, such as PlotReport extension, PrintReport extension.
    dnn_model.disable_extensions()
    if pruning_enabled:
        # Try to prune the trial for each 3 epoch
        pruning_ext = optuna.integration.ChainerPruningExtension(
            trial, 'val/main/loss', (3, 'epoch')
        )
        dnn_model.set_trainer_extension('pruning', pruning_ext)
    _, loss = dnn_model.fit(train_data)
    dnn_model.save(out_dir)
    # Delete intermediate files to save disk space
    dnn_model.clean_artifacts(out_dir)
    return loss


def main():
    parser = get_parser()
    args = parser.parse_args()
    normal_file = utils.to_absolute(args.normal)
    if not normal_file.exists():
        print("{} does not exist.".format(normal_file))
        exit(1)

    out_dir = utils.mkdir(args.output_dir)
    # Read CSV data and convert it to pandas.DataFrame
    normal_data = SWaTData(normal_file)
    normal_df = normal_data.read()
    train_data = normal_df[[normal_data.continuous_columns, *normal_data.discrete_columns]]

    # Study file is a database file of Optuna
    study_file = "sqlite:///" + str(out_dir / 'optimize.db')

    print("----- Train data -----")
    print("File: {}".format(normal_file))
    print("Length: {}".format(len(train_data)))
    print("Number of features: {}".format(len(train_data.columns)))
    print("----- Optimizing info -----")
    print("Device ID: {}".format(args.device))
    print("Output dir: {}".format(out_dir))
    print("Trials: {}".format(args.trials))
    print("Study: {}".format(study_file))

    # Optimize the params with given data
    print("----- Start to optimize -----")
    median_pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        study_file,
        study_name="example_subtilin",
        load_if_exists=args.resume,
        pruner=median_pruner
    )
    study.optimize(lambda x: train(x, train_data, out_dir, args.device, args.prune), args.trials)
    print("----- End -----")


if __name__ == "__main__":
    main()
