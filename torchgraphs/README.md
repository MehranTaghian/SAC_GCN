# Torchgraphs

A PyTorch library for Graph Convolutional Networks.

## Requirements and installation

Torchgraphs is developed in Python 3.7 and depende on PyTorch 1.1.
It is suggested but not required to install PyTorch beforehand, to correctly match the hardware capabilities, 
see the official installation [instruction](https://pytorch.org/).

All requirements listed in [requirements.txt](./requirements.txt) will be installed automatically when running:
```bash
pip install https://github.com/baldassarreFe/torchgraphs
```

To develop the library itself, a conda environment with additional dependencies is provided, 
as well as a test suite for [pytest](https://pytest.org): 
```bash
git clone https://github.com/baldassarreFe/torchgraphs
cd torchgraphs

ENV_NAME=torchgraphs_env
conda env create -n "${ENV_NAME}" -f conda.yaml 
conda activate "${ENV_NAME}"

pip install --editable .
pytest
```