# LANfactory

![PyPI](https://img.shields.io/pypi/v/lanfactory)
![PyPI_dl](https://img.shields.io/pypi/dm/lanfactory)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Lightweight python package to help with training [LANs](https://elifesciences.org/articles/65074) (Likelihood approximation networks). 

Please find the original [documentation here](https://alexanderfengler.github.io/LANfactory/).

### Quick Start

The `LANfactory` package is a light-weight convenience package for training `likelihood approximation networks` (LANs) in torch (or keras), 
starting from supplied training data.

[LANs](https://elifesciences.org/articles/65074), although more general in potential scope of applications, were conceived in the context of sequential sampling modeling
to account for cognitive processes giving rise to *choice* and *reaction time* data in *n-alternative forced choice experiments* commonly encountered in the cognitive sciences.

For a basic tutorial on how to use the `LANfactory` package, please refer to the [basic tutorial notebook](docs/basic_tutorial/basic_tutorial.ipynb).

In this quick tutorial we will use the [`ssms`](https://github.com/AlexanderFengler/ssm_simulators) package to generate our training data using such a sequential sampling model (SSM). The use is in no way bound to utilize the `ssms` package.

#### Install

To install the `ssms` package type,

`pip install ssm-simulators`

To install the `LANfactory` package type,

`pip install lanfactory`

Necessary dependency should be installed automatically in the process.

### Basic Tutorial

Check the basic tutorial [here](docs/basic_tutorial/basic_tutorial.ipynb).
    
### TorchMLP to ONNX Converter

Once you have trained your model, you can convert it to the ONNX format using the `transform_onnx.py` script.

The `transform_onnx.py` script converts a TorchMLP model to the ONNX format. It takes a network configuration file (in pickle format), a state dictionary file (Torch model weights), the size of the input tensor, and the desired output ONNX file path.

### Usage

```python onnx/transform_onnx.py <network_config_file> <state_dict_file> <input_shape> <output_onnx_file>```

Replace the placeholders with the appropriate values:

- <network_config_file>: Path to the pickle file containing the network configuration.
- <state_dict_file>: Path to the file containing the state dictionary of the model.
- <input_shape>: The size of the input tensor for the model (integer).
- <output_onnx_file>: Path to the output ONNX file.

For example:

```
python onnx/transform_onnx.py '0d9f0e94175b11eca9e93cecef057438_lca_no_bias_4_torch__network_config.pickle' '0d9f0e94175b11eca9e93cecef057438_lca_no_bias_4_torch_state_dict.pt' 11 'lca_no_bias_4_torch.onnx'
```
This onnx file can be used directly with the [`HSSM`](https://github.com/lnccbrown/HSSM) package. 

We hope this package may be helpful in case you attempt to train [LANs](https://elifesciences.org/articles/65074) for your own research.

#### END

