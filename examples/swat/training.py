import argparse
from pathlib import Path

from deep_sentinel.models import dnn
from deep_sentinel import utils

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
    parser = argparse.ArgumentParser(description="Sample implementation to train the model"
                                                 " with SWaT")
    parser.add_argument("--normal",
                        type=Path,
                        required=True,
                        help="Path to normal data")
    parser.add_argument("-o",
                        "--output-dir",
                        type=Path,
                        required=True,
                        help="Path to output dir")
    train_params = parser.add_argument_group("Train params")
    train_params.add_argument("-b",
                              "--batch-size",
                              type=int,
                              default=16,
                              help="Minibatch size")
    train_params.add_argument("-d",
                              "--device",
                              type=str,
                              default="-1",
                              help="Device ID to use (negative value indicate CPU)")
    train_params.add_argument("-e",
                              "--max-epoch",
                              type=int,
                              default=20,
                              help="Max epoch")
    train_params.add_argument("-r",
                              "--dropout-ratio",
                              type=float,
                              default=0.5,
                              help="Dropout ratio")
    model_params = parser.add_argument_group("Model params")
    model_params.add_argument("-n",
                              "--n-units",
                              type=int,
                              default=64,
                              help="Hidden size")
    model_params.add_argument("-l",
                              "--lstm-stack",
                              type=int,
                              default=1,
                              help="Number of stacked LSTM unit")
    model_params.add_argument("-p",
                              "--bprop-length",
                              type=int,
                              default=1000,
                              help="Update parameters once with the specified number of data")
    model_params.add_argument("-a",
                              "--activation",
                              type=str,
                              choices=activation_funcs,
                              default=activation_funcs[0],
                              help="Which activation function to use "
                                   "({} are available)".format(activation_funcs))
    model_params.add_argument("-g",
                              "--gmm-classes",
                              type=int,
                              default=1,
                              help="Number of GMM classes")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    normal_file = utils.to_absolute(args.normal)
    if not normal_file.exists():
        print("{} does not exist.".format(normal_file))
        exit(1)

    out_dir = utils.mkdir(args.output_dir)

    print("----- Model Params -----")
    print("Minibatch: {}".format(args.batch_size))
    print("N units: {}".format(args.n_units))
    print("LSTM Stack: {}".format(args.lstm_stack))
    print("Back propagation length: {}".format(args.bprop_length))
    print("Activation Function: {}".format(args.activation))
    print("GMM Classes: {}".format(args.gmm_classes))

    # Create model instance
    dnn_model = dnn.DNN(
        batch_size=args.batch_size,
        output_dir=out_dir,
        max_epoch=args.max_epoch,
        device=args.device,
        n_units=args.n_units,
        lstm_stack=args.lstm_stack,
        dropout_ratio=args.dropout_ratio,
        activation=args.activation,
        bprop_length=args.bprop_length,
        gmm_classes=args.gmm_classes
    )

    normal_data = SWaTData(normal_file)
    normal_df = normal_data.read()

    print("----- Train data -----")
    print("File: {}".format(normal_file))
    print("Length: {}".format(len(normal_df)))
    print("Number of features: {}".format(len(normal_df.columns)))

    print("----- Train info -----")
    print("Max epoch: {}".format(args.max_epoch))
    print("Device ID: {}".format(args.device))
    print("Dropout ratio: {}".format(args.dropout_ratio))
    print("Output dir: {}".format(out_dir))

    # Train the model with given data
    print("----- Start to train -----")
    # Learn normal data without labels
    dnn_model.fit(normal_df[[*normal_data.continuous_columns, *normal_data.discrete_columns]])

    # Dump the weights of the model whose performance is the best.
    saved_model = dnn_model.save(out_dir)
    print("Save as: {}".format(saved_model))
    print("----- End -----")


if __name__ == '__main__':
    main()
