import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")

import matplotlib.pyplot as plt

from deep_sentinel import utils
from deep_sentinel.models import dnn


def read_simulations(data_dir: 'Path') -> 'np.ndarray':
    simulations_dir = data_dir / "1" / "simulations"
    csv_files = list(simulations_dir.glob("*.csv"))
    # Each CSV's dimensions is (Steps, Features)
    csv_contents = [pd.read_csv(str(f)).values for f in csv_files]
    # Stack them into one ndarray. The shape becomes (Trials, Steps, Features)
    return np.stack(csv_contents)


def plot_results(sampled: 'np.ndarray', ground_truth: 'np.ndarray'):
    titles = ["Ground truth", "Sampled values"]
    colors = ["m", "g"]
    steps = sampled.shape[1]
    x_axis = range(1, steps + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharey="all", sharex="all")
    for i, arr in enumerate([ground_truth, sampled]):
        title = titles[i]
        color = colors[i]
        if arr.shape[1] > steps:
            arr = arr[:, :steps]
        _mean = np.mean(arr, axis=0)
        _std = np.std(arr, axis=0)
        _min = _mean - _std
        _max = _mean + _std
        # Plot typical actual values.
        _example = arr[0]
        axes[i].set_title(title)
        axes[i].plot(x_axis, _mean, '{}-.'.format(color))
        axes[i].plot(x_axis, _min, '{}:'.format(color))
        axes[i].plot(x_axis, _max, '{}:'.format(color))
        axes[i].plot(x_axis, _example, '{}-'.format(color))
    return fig


def get_parser():
    parser = argparse.ArgumentParser(description="Sample implementation to train the model"
                                                 " with subtilin production data")
    parser.add_argument("-i",
                        "--input-dir",
                        type=Path,
                        required=True,
                        help="Path to dir including train data")
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
    in_dir = utils.to_absolute(args.input_dir)
    if not in_dir.exists():
        print("{} does not exist.".format(in_dir))
        exit(1)

    out_dir = utils.mkdir(args.output_dir)

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

    # Initialize with given seed data
    seed_csv = in_dir / "1" / "subtilin-seed.csv"
    if not seed_csv.exists():
        print("{} does not exist. Please generate seed data.".format(seed_csv))
        exit(1)

    # Read seed data
    seed_data = pd.read_csv(str(seed_csv), index_col=[0], header=[0])
    print("----- Seed data -----")
    print("File: {}".format(seed_csv))
    print("Length: {}".format(len(seed_data)))
    print("----- Start to sample -----")
    steps = args.steps
    trials = args.trials
    print("# Trials: {}".format(trials))
    print("# Steps: {}".format(steps))

    # Initialize the internal state of DNN model
    dnn_model.initialize_with(seed_data)

    # Sampling values from estimated distributions
    sampled = dnn_model.sample(steps=steps, trials=trials)

    # Obtain all simulated results
    ground_truth = read_simulations(in_dir)

    # Shape of `sampled` and `ground_truth` are (Trials, Steps, Features)
    # Extract [SpaS] (a concentration of SpaS) only to plot
    figure = plot_results(sampled[:, :, -1], ground_truth[:, :, -1])
    print("Save as: {}".format(str(out_dir / 'sampled.png')))
    figure.savefig(str(out_dir / 'sampled.png'))
    print("----- End -----")


if __name__ == "__main__":
    main()
