import os
import numpy as np
import os.path as osp

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
metadata_path = '/home/chandramouli/Downloads/cifar-100-python/meta' # change this path`\
metadata = unpickle(metadata_path)
superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))
data_pre_path = '/home/chandramouli/Downloads/cifar-100-python/' # change this path
# File paths
data_train_path = data_pre_path + 'train'
data_test_path = data_pre_path + 'test'
# Read dictionary
data_train_dict = unpickle(data_train_path)
data_test_dict = unpickle(data_test_path)
# Get data (change the coarse_labels if you want to use the 100 classes)
del data_train_dict[b'data']
coarse_label_train = np.array(data_train_dict[b'coarse_labels'])
fine_label_train = np.array(data_train_dict[b'fine_labels'])
filename_train = data_train_dict[b'filenames']
data_test = data_test_dict[b'data']
filename_test = data_test_dict[b'filenames']
label_test = np.array(data_test_dict[b'coarse_labels'])
data_train_dict[b'filenames'] = [x.decode("utf-8")for x in data_train_dict[b'filenames']]
coarse_labels = dict(zip(data_train_dict[b'filenames'], data_train_dict[b'coarse_labels']))
# print(coarse_labels)