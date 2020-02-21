# DeepSentinel:  Anomaly detector based on a deep neural network

Learning Stochastic Hybrid System behavior by Deep Neural Networks (DNN). This implementation is based on the model proposed by Inoue et al\[1\].

## Status

Currently, DeepSentinel is under development. A supposed user is a researcher who want to try a new technology.

## Environment

We confirm this package on the following platforms.

- Linux
  - Ubuntu 18.04 LTS 64bit (Recommended)
  - Ubuntu 16.04 LTS 64bit
  - CentOS 7 64bit 
- macOS
  - Catalina 10.15.x
  - Mojave 10.14.x

And these software. Newer version of Chainer (and CuPy) is recommended to use.

- Python 3.5 or later
  - Chainer 5.3.0 or later
  - CuPy (Optional) 5.3.0 or later
- CUDA (supported by CuPy)
  - 9.2
  - 10.0
  - 10.1 (CuPy 6.0.0 or later)
  - 10.2 (CuPy 7.2.0 or later)

## Installation

DeepSentinel is available to be installed by pip from GitHub.

```bash
# Install latest version of DeepSentinel
$ pip install git+https://github.com/yoriyuki-aist/DeepSentinel
# Specify the version
$ pip install git+https://github.com/yoriyuki-aist/DeepSentinel@3.0.0
```

### (Optional) Working with GPU

Installing CUDA is **strongly recommended**. It takes a too long time to complete the learning or inference without GPU support.  
CUDA is not supported on macOS Mojave or later. Use Ubuntu or CentOS instead.

See https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation to get more details.

In addition, please install CuPy manually. It is recommended to install CuPy built with a specific version of CUDA and cuDNN.

```
# CUDA 9.2
$ pip isntall cupy-cuda92
# CUDA 10.0
$ pip install cupy-cuda100
```

As an alternative, building CuPy with your own CUDA librarieis is also available. Please read [the installation guides of CuPy](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy-from-source) to get more detail. 

## Usage

Please see the [examples](examples) to get the complete codes.

- One-step-ahead prediction
  - [An example using Subtilin Production model](examples/subtilin)
- Anomaly detection
  - [An example using Secure Water Treatment system](examples/swat)

Here is a simple way to learn a sine curve.

```bash
from deep_sentinel.models import dnn

model = dnn.DNN(
    # Size of mini batch.
    batch_size=16,
    # GPU ID to use (a negative value indicates CPU).
    device=-1,
    # Number of dimension of hidden state.
    n_units=64,
    # Number of stack of LSTM.
    lstm_stack=1,
    # Drop out ratio (0 <= N < 1).
    dropout_ratio=0.0,
    # Activation function to use. `sigmoid` and `relu` are available.
    activation='sigmoid',
    # Back propagation length.
    # Update parameters once with the specified number of data.
    bprop_length=1000,
    # Max epoch to learn.
    # This model supports early stopping.
    # Learning may stop before the learning iteration reaches this value.
    max_epoch=20,
    # Path to save logs and intermediate model files
    output_dir='/path/to/dir',
    # Number of distribution to use.
    # Gaussian Micture Model is supported.
    # When a value of 2 or more is specified, GMM is used for learning and inference.
    gmm_classes=1
)

# Prepare data to learn
import numpy as np
import pandas as pd
sin_curve = np.sin(np.linspace(0, 1000, 10000))
data = pd.DataFrame(sin_curve)
# If you want to use a dicrete value (categorical data),
# you should make its column type as `category` as follows.
# 
# discrete_values = np.random.randint(0, 2, (10000,))
# data = pd.DataFrame({
#     'sin': sin_curve,
#     'randint': discrete_values,
# })
# data['randint'] = data['randint'].astype('category')

# Train the model
# Matplotlib is required to plot the loss value.
import matplotlib as mpl
mpl.use("Agg")
dnn_model.fit(data)

# Initialize with seed data
seed_data = pd.DataFrame(sin_curve[:100])
dnn_model.initialize_with(seed_data)

# MCMC like sampling from given seed data
steps = 100
trials = 1
dnn_sampled = dnn_model.sample(steps, trials)

# Plot the results
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
result_df = pd.DataFrame({
    "Ground truth": sin_curve[100:200],
    "DNN": dnn_sampled.reshape(-1),
    "SSM": ssm_sampled.reshape(-1),
    "HMM": hmm_sampled.reshape(-1),
})
result_df.plot(ax=ax) 
fig.savefig('results.png')
```

## Development

Use pipenv to lock the packages and manage a virtual environment.

```bash
$ pipenv install --dev
```

Running tests by pytest.

```bash
$ pipenv run tests
```

Testing with multiple versions of Python and Chainer is supported by tox.
Python 3.5, 3.6, 3.7, and 3.8 must be available.

```bash
# Test all environments
$ pipenv run tox -p all
# Test in specified version of Python and Chainer
$ pipenv run tox -e py35-chainer5
```

## License

MIT License

## Copyright

Copyright (C) 2017-2019: National Institute of Advanced Industrial Science and Technology (AIST)

## References

\[1\]: Jun Inoue, Yoriyuki Yamagata, Yuqi Chen, Christopher M Poskitt, and Jun Sun. Anomaly detection for a water treatment system using unsupervised machine learning. In2017 IEEE International Conferenceon Data Mining Workshops (ICDMW), pages 1058–1065. IEEE, 2017.
