# Are wider nets better given the same number of parameters?

This repository contains the code used for the experiments in the following paper:

["Are wider nets better given the same number of parameters?"](https://arxiv.org/abs/2010.14495)<br/>
Anna Golubeva, Behnam Neyshabur, Guy Gur-Ari.<br/>
International Conference on Learning Representations (ICLR), 2021.

**Disclaimer**: this is not an official Google product.

## Getting Started
Clone this repo, then install all dependencies:
```
pip install -r requirements.txt
```
The code was tested with Python 3.6.8.

## Code Organization
Below is a description of the major sections of the code base. Run `python main.py --help` for a complete description of flags and hyperparameters.

### Datasets
This code supports the following datasets: CIFAR-10, CIFAR-100, MNIST, SVHN.
All datasets will download automatically.


### Models
We consider two types of models: MLP and ResNet18. 

 - `MLP`: MLP (i.e., fully-connected feed-forward achitecture) with 1 hidden layer for MNIST experiments
 - `ResNet18`: models with ResNet18 architecture from [this repo](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py) for CIFAR-10, CIFAR-100 and SVHN

Use the file `generate_arg.py` in the respective folder to set the experiment parameters. Calling
```
python generate_args.py
```
will print out commands to start the main script from the shell (locally).
For ResNet18 experiments, it will also dump a dictionary specifying all job parameters into a json file, which is convenient to use if submitting jobs to a cluster or to the cloud.


## Citation
If you use this code for your research, please cite our paper
["Are wider nets better given the same number of parameters?"](https://arxiv.org/abs/2010.14495).


