# Anomaly detection of SWaT system

This is an example of anomaly detection of SWaT system\[1\] using DeepSentinel.

**NOTE**: This example require a lot of compute resources. It is recommended to use GPU which has large GPU memory.

## Installation

```bash
$ cd path/to/swat
$ pipenv install
```

This example require some additional modules.

```bash
$ pipenv run pip install xlrd sklearn
```

## Dataset

You can request to download the dataset from [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/).

## Usage

Please see the help messages of the scripts for more detail.

### Training

SWaT dataset contains two types of data, normal data and attack data.
Train the model by normal data in this approach.

```bash
$ pipenv run python training.py \
    --normal path/to/SWaT_Dataset_Normal_v1.xlsx \
    -o output/default
```

This step will generate following files.

```bash
$ tree ./output/default
./output/default
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

### Evaluate

Evaluate the model by performing anomaly detection.

```bash
$ pipenv run python evaluate.py --attack path/to/SWaT_Dataset_Attack_v1.xlsx \
    -o output/default \
    --model output/default/dnn-model
```

The results will be saved as `output/default/metrics_summary.csv` and `output/default/metrics_detail.csv`.

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

## References

\[1\]: Jonathan  Goh,  Sridhar  Adepu,  Khurum  Nazir  Junejo,  and  AdityaMathur. A dataset to support research in the design of secure water treatment systems. InProceedings in the 11th International Conferenceon Critical Information Infrastructures Security, number October, 2016.