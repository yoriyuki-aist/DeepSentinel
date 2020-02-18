# Subtilin production model

This is a example of learning the behavior of stochastic hybrid system (Subtilin production in Bacillus Subtilis\[1\]).

## Installation

```bash
$ cd path/to/examples/subtilin
$ pipenv install
```

## Usage

Please see the help messages of the scripts for more detail.

### Generate training dataset

First, generate sample data to learn. This step will require a lot of computational resources and time.
We recommend you to run this script on the computer which has a large number of faster CPU cores. 

Seed data is a short length of time-series (\~1000). It is used as the initial data to initialize the internal state of DNN model.
Trial data is a ground truth. It is generated for each seed data.

We recommend that use the number of seeds as 1 since the limitation of resources.


```bash
$ pipenv run python simulate.py \
    -o dataset/ \
    --train-length 100000 \
    --total-seeds 1 \
    --seed-length 1000 \
    --total-trials 1000 \
    --trial-length 300
```

The script will create following structures.

```bash
$ tree -L 3 ./dataset/
./dataset/
├── 1
│   ├── simulations
│   │   ├── subtilin-1.csv
│   │   ├── subtilin-2.csv
│   │   ├── ......
│   │   ├── subtilin-98.csv
│   │   └── subtilin-99.csv
│   └── subtilin-seed.csv
└── subtilin-training.csv
```

### Training

Training model with default params:

```bash
$ pipenv run python training.py \
    -i dataset/ \
    -o output/default
```

This step will generate following files.

```bash
$ tree ./output/default
./output/sub
├── dnn-model
├── dnn-model-meta
├── log
├── loss.png
├── snapshot-epoch-1
├── snapshot-epoch-2
├── ......
├── snapshot-epoch-19
└── snapshot-epoch-20
```

### Sampling values

Sample values from the end of seed data. This script require the model file to use.

```bash
$ pipenv run python sampling.py \
    -i dataset/ \
    -o output/default \
    -m output/default/dnn-model
```

### Optimizing hyper parameters

This step require a additional module, [Optuna](https://github.com/optuna/optuna).

```bash
$ pipenv run pip install "optuna==1.1.0"
```

Specify the number of trials. Note that it also includes successful, failed, and pruned trials.

```bash
$ pipenv run python optimize.py \
    -i dataset/ \
    -o output/opt \
    -t 300
```

## Note

### ChainerX

DeepSentinel supports [ChainerX](https://docs.chainer.org/en/stable/chainerx/index.html) on training step. However, sampling feature does **not** support it.
You can use ChainerX as follows.

```bash
# Do not use ChainerX (CPU)
$ pipenv run python training.py --device -1
# Do not use ChainerX (GPU)
$ pipenv run python training.py --device 0
# Use ChainerX with CPU
$ pipenv run python training.py --device native
# Use ChainerX with GPU
$ pipenv run python training.py --device cuda:0
```

ChainerX is available on hyper parameter optimization, too.

## License

See [LICENSE](../../LICENSE.md)

## References

\[1\]: Jianghai Hu, Wei-chung Wu, and Shankar Sastry. Modeling Subtilin production in Bacillus Subtilis using stochastic hybrid systems.  InInternational Workshop on Hybrid Systems: Computation and Control,pages 417–431. Springer, 2004 (March), 2014
