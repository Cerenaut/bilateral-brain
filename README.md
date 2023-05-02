# Deep Learning in a Bilateral Brain

Original RFR [here](https://wba-initiative.org/en/research/rfr/rfr-left-and-right-nn/)

Preprint [here](https://arxiv.org/abs/2209.06862)

## Overview

The brains of all bilaterally symmetric animals on Earth are divided into left and right hemispheres. It is a remarkably conserved feature across species, indicating its importance for intelligence. The anatomy and functionality of the hemispheres have a large degree of overlap, but they specialise to possess different attributes. The most likely explanation for the emergent specialisation, is small differences in parameterisation of the substrate. For example, a higher rate of synaptic plasticity in one hemisphere, different relative layer sizes and different connectivity patterns within and across layers. The biological parameterisation could be equivalent to hyperparameters of an AI Machine Learning algorithm.

There could be great benefits to understanding and mimicking this pervasive design feature of biological intelligence by building bilateral ML algorithms – two parallel systems with different specialities. To our knowledge, this has never been explored. It has the potential to confer significant advantages and is an exciting prospect. The way to approach it is an open question, we can be inspired and guided by neuroscience models.

## Objective

The objective is to build a bilateral Machine Learning model with hemispheres resembling the biological counterparts. This will form the basis for further explorations of bilateralism, including interplay with other brain and neurotransmitter systems.


# Getting started

## Terminology

In this project, we use a backbone (e.g. resnet or vgg type architectures) to create larger networks (either 1 hemisphere or 2) with one or two classifier heads.

- **Fine/Coarse**: This project is designed for hierarchical datasets such as CIFAR100, where each image has a `fine` and a `coarse` label. `Fine` is the narrow/specific description, such as dolphin, and `Coarse` is the broad/general description, such as sea creature
- **Architecture**: The backbone used i.e. the architecture for a single hemisphere
- **Single/Dual Head**: The number of heads. Each head is a classifier, and can be trained on either fine labels or coarse labels
- **Macro-architecture**: The whole network, including hemispheres and heads


## The main idea
The main idea is to:
1) Train individual hemispheres in a way that makes them specialized
So one hemisphere is trained on fine labels, the other with coarse.

2) Then the pre-trained hemispheres are combined into a bilateral architecture (i.e. two hemispheres), with two heads. One head is for fine labels, the other for coarse.
The output of each hemisphere are concatenated to create one set of features that are fed into each head. 
The hemispheres are pre-trained in step (1), and so here they are frozen (they will not be trained); only the heads are trained and tested.


## Guide to the folder structure:
- `arch_single_head`: Train/Test one single hemisphere, with one head
- `arch_dual_head`: Train/Test a macro-architecture consisting of optionally one or two hemispheres, with two heads


## Preparing the data

Download [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) for python, and put it in your chosen folder.

Modify the paths in `data_scripts/prepare_cifar.py` (at the top of the file), and run the script to create the fine and coarse dataset folders.

Each folder has a `config.yaml` which you can use configure the experiment.
In particular, set the path to dataset and checkpoints there.

In each folder you have:

- `trainer.py`: used to train and validate
- `test.py`: used to test

## Run the system
The easiest way to run the system in the stereotypical way (as in the section 'The main idea' above), is to use the `train_system.py` script. It enables you to run several seeds for each single hemisphere, and then run several seeds on the whole bilateral architecture also. It is also quite configurable i.e. different backbones.
The script depends the base configs and modifies them.

You can have finer level control, and do different variations, by running `trainer.py` in `arch_single_head` and in `arch_dual_head` and creating new config files as required.

Examples of how to do that are given below, first in the context of the stereotypical scenario


## Example 1: bilateral architecture with specialization

As an example, here are step by step instructions to train Left on fine classes and Right on coarse classes, then put them together into bilateral architecture and train on all class types.

This will specialize the left for fine and the right for coarse classes, using supervised training. As in the paper, you can then apply additional asymmetries to enhance specializations such as sparsity or having hemispheres with asymmetric layer widths.

### 1. Train Left on specific/fine labels

Go to `arch_single_head` folder.

Configure the experiment by modifying the `config.yaml` file in the `config/` folder, to:

- use specific labels (`config/dataset/`), and
- change the name of the experiments (`config/exp_name`) 
- update `mode` appropriately: `fine` for fine labels, `coarse` for coarse labels or `feature` for the features rather than a prediction
- update `arch` to set the backbone architecture
- set `evaluate` to `True` if you want to also test the accuracy after training
- see `config.yaml` for explanations of other parameters

Then run:
``python trainer.py``


#### 2. Train Right on general/coarse labels 

Then do it all again on coarse labels to train the Right hemisphere.

#### 3. Create bilateral architecture

Go to `arch_dual_heads` folder.
Configure to use the appropriate checkpoints to load the Left and Right hemispheres.

- update `mode` to `both`. In this case it determins where the output is taken from.
- update `macro_arch` to `bilateral`
- set the `farch` and `carch` to the appropriate backbones of `fine` and `coarse` hemispheres respectively
- set them to be fronzen with `ffreeze` and `cfreeze`
- set the dataset paths in the config to point to the `fine` folders and include the path to the raw files.
The DataModule will then pick up `fine` labels from the image names and the `coarse` labels from the raw files.
For example:

```
dataset:
  train_dir: ../datasets/CIFAR100/train/fine
  val_dir: ../datasets/CIFAR100/test/fine
  test_dir: ../datasets/CIFAR100/test/fine
```

- see `config.yaml` for explanations of other parameters
- set `evaluate` to True to get test accuracy

Then run:
``python trainer.py``


## Example 2: How to train a single hemisphere on two labels

Use `\arch_dual_heads`. 
- this time set the config hparam `macro_arch` to `unilateral`
- it will use all the `fine` hparams e.g. farch and ffreeze, to specify this one hemisphere
- set `ffreeze` to False

Then run:
``python trainer.py``
