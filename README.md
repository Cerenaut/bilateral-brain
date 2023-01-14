# bilateral-brain

Guide to the folder structure:

- classification:
Train one single hemispheres at a time.

- hemishperes:
Combine the hemispheres and train and test only the last layers.


**Preparation**

Download [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) for python, and put it in your chosen folder.

Modify the paths in `prepare_cifar.py` (at the top of the file), and run the script to create the fine and coarse dataset folders.

Each folder has a `config.yaml` which you can use configure the experiment.
In particular, set the path to dataset and checkpoints there.

In each folder you have:

- `trainer.py`: used to train and validate
- `test.py`: used to test

**Getting started**

As an example, here are step by step instructions to train Left on specific classes and Right on general classes, then put them together into bilateral architecture and train on all class types.

1. Train Left on specific classes

Go to `classification` folder.
Configure to use specific labels and run:
``python trainer.py``

Get the test accuracy with `python test.py`.
Note the name of the checkpoint.


Configure to use general labels and run:
``python trainer.py``

Get the test accuracy with `python test.py`.
Note the name of the checkpoint.


Go to `hemispheres` folder.
Configure to use the appropriate checkpoints.

Run ``python trainer.py``

Get the test accuracy with `python test.py`.


# TODO remove hardcoding of paths in test.py
CKPT_PATH
TEST_FOLDER