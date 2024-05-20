# MLPs Learn In-Context
This repository contains code that reproduces experiments from our paper [MLPs Learn In-Context](https://www.google.com)

## Installation

We use Python 3.10.12.

To install the requisite Python packages, run
```sh
pip install -r requirements.txt
```
By default, this will install CPU-only Jax. For GPU-enabled Jax, run
```sh
pip install -U "jax[cuda12]"
```
Depending on your machine, you may need additional or alternative packages for GPU-enabled Jax. Please consult the [Jax installation instructions](https://jax.readthedocs.io/en/latest/installation.html) for details.


## Running the code
Code organization:
* `experiment/`: experiments and plotting
* `model/`: model implementations
* `task/`: task implementations
* `train`: training routines

Experiments are organized per file. For example, `12_icl_clean.py` will plot results for ICL regression and classification. Each experiment file expects results to be present in `experiment/remote`, which contains scripts that generate the results, and are intended to be run on a compute cluster. Running the corresponding `run.py` script for each sub-experiment will generate the corresponding results.

Experiment files are formatted as [Jupyter code cells](https://code.visualstudio.com/docs/python/jupyter-support-py#_export-a-jupyter-notebook). One way of running these files interactively is through the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) in VSCode. However, because the formatting occurs through comment-based delimiters, these files may also be run as standard Python files.
