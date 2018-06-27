# DeepSentinel:  Anomaly detector based on a deep neural network

Copyright (C) 2017, 2018: National Institute of Advanced Industrial Science and Technology (AIST)

## Status
Currently, DeepSentinel is under developement.  A supposed user is a researcher who want to try a new technology.

## Download
Clone from the git repo.

## Installation
No installation procedure is provided yet.  DeepSentinel depeneds on

- GPGPU and CUDA (optional but highly recommended)
- Chainer
- pandas

## Usage

Let Normal.xslx be a normal log data in Excel format, and Attack.xslx be a attack data.

First, we need proprocessing:
```shell
python preprocess.py Normal.xslx
python preprocess.py --normal Normal Attack.xslx
```
You need`output` directory under your working directory.

Next, we learn the system model:
```shell
python train.py -g 0 -i 10 Normal
```

Finally, we calculate outlier factors:
```shell
python evaluate.py Normal-1-100-dropout-True-sigmoid-10-lstms.npz Attack score.csv
```