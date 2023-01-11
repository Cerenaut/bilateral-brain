# bilateral-brain

Guide to the folder structure:

**Hemishperes:** 
Combine the hemispheres and train and test only the last layers.

**Classification:** 
train one single hemispheres at a time

Preparation
Download CIFAR100 and run the prepare_cifar.py script to create the broad and narrow datasets.

For example, here are step by step instructions to train Left on specific classes and Right on general classes, then put them together into bilateral architecture and train on all class types.

1. Train Left on specific classes

First modify the path to the data and split file on your local machine, in `trainer.py` (lines 80 and 81), and `datamodule.py` (line 139, 203, 207).

@chandra: 
- what is the split file? 
- what are the instructions for downloading CIFAR100. Just get it from cifar website and put in the data folder?

Then run this:
`python trainer.py` on Left, specifying 'specific' classes ... ??? how!?
and then again on Right, specifying 'general' classes ... ??? how!?

@chandra
- The hemisphere model is saved in a checkpoint at ???? 
- How do you specify specific classes for one, and general classes for teh next

@chandra how do you specify the checkpoints to load?


General questions (@chandra)
- is 'default_root_dir' used for anything? I see it is an argument sent to supervised module
- in each folder, are `trainer.py` and `test.py` used for the obvious? i.e. trainer to train a model with validation, providing a checkpoint, and then you can load the checkpoint in test to give an accuracy on held out test set?