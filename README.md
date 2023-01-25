# Deep Learning in a Bilateral Brain

Original RFR [here](https://wba-initiative.org/en/research/rfr/rfr-left-and-right-nn/)

Preprint [here](https://arxiv.org/abs/2209.06862)

## Overview

The brains of all bilaterally symmetric animals on Earth are divided into left and right hemispheres. It is a remarkably conserved feature across species, indicating its importance for intelligence. The anatomy and functionality of the hemispheres have a large degree of overlap, but they specialise to possess different attributes. The most likely explanation for the emergent specialisation, is small differences in parameterisation of the substrate. For example, a higher rate of synaptic plasticity in one hemisphere, different relative layer sizes and different connectivity patterns within and across layers. The biological parameterisation could be equivalent to hyperparameters of an AI Machine Learning algorithm.

There could be great benefits to understanding and mimicking this pervasive design feature of biological intelligence by building bilateral ML algorithms – two parallel systems with different specialities. To our knowledge, this has never been explored. It has the potential to confer significant advantages and is an exciting prospect. The way to approach it is an open question, we can be inspired and guided by neuroscience models.

## Objective

The objective is to build a bilateral Machine Learning model with hemispheres resembling the biological counterparts. This will form the basis for further explorations of bilateralism, including interplay with other brain and neurotransmitter systems.


# Getting started

### Guide to the folder structure:

- classification:
Train one single hemispheres at a time.

- hemishperes:
Combine the hemispheres and train and test only the last layers.


### Preparing the data

Download [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) for python, and put it in your chosen folder.

Modify the paths in `prepare_cifar.py` (at the top of the file), and run the script to create the fine and coarse dataset folders.

Each folder has a `config.yaml` which you can use configure the experiment.
In particular, set the path to dataset and checkpoints there.

In each folder you have:

- `trainer.py`: used to train and validate
- `test.py`: used to test


### Example of bilateral architecture with specialization

As an example, here are step by step instructions to train Left on specific classes and Right on general classes, then put them together into bilateral architecture and train on all class types.

This will specialize the left for specific and the right for general classes, using supervised training. As in the paper, you can then apply additional asymmetries to enhance specializations such as sparsity or having hemispheres with asymmetric layer widths.

#### 1. Train Left on specific/narrow classes (fine labels)

Go to `classification` folder.

Configure the experiment by modifying the `config.yaml` file in the `config/` folder, to:

- use specific labels (`config/dataset/`), and
- change the name of the experiments (`config/logger/name`) 

Then run:
``python trainer.py``

You can get the test accuracy with `python test.py`.
First set the correct checkpoint path in the config.

#### 2. Train Right on general/broad classes (coarse labels)

Then do it all again on general labels to train the Right hemisphere.

#### 3. Create bilateral architecture

Go to `hemispheres` folder.
Configure to use the appropriate checkpoints to load the Left and Right hemispheres.

This will assemble the hemispheres and add a MLP head.

Run ``python trainer.py``

Get the test accuracy with `python test.py`.
