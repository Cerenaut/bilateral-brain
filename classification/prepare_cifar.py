import os
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
metadata_path = '/path_cifar-100-dataset' # change this path`\
metadata = unpickle(metadata_path)
superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))

data_pre_path = '/path_cifar-100-dataset' # change this path
# File paths
data_train_path = data_pre_path + 'train'
data_test_path = data_pre_path + 'test'
# Read dictionary
data_train_dict = unpickle(data_train_path)
data_test_dict = unpickle(data_test_path)
# Get data (change the coarse_labels if you want to use the 100 classes)
print(data_train_dict.keys())
data_train = data_train_dict[b'data']
coarse_label_train = np.array(data_train_dict[b'coarse_labels'])
fine_label_train = np.array(data_train_dict[b'fine_labels'])
filename_train = data_train_dict[b'filenames']
data_test = data_test_dict[b'data']
filename_test = data_test_dict[b'filenames']
label_test = np.array(data_test_dict[b'fine_labels'])
print(fine_label_train)
print(coarse_label_train)
dir = '/home/chandramouli/Documents/kaggle/CIFAR-100'
# base_dir = osp.join(dir, 'test')
# os.makedirs(base_dir, exist_ok=True)

# for i in tqdm(range(data_train.shape[0])):
#     if not osp.exists(osp.join(base_dir, str(label_train[i]))):
#         os.makedirs(osp.join(base_dir, str(label_train[i])))
#     r = data_train[i][:1024].reshape(32, 32)
#     g = data_train[i][1024:2048].reshape(32, 32)
#     b = data_train[i][2048:].reshape(32, 32)
#     img = np.stack([b, g, r]).transpose((1, 2, 0))
#     cv2.imwrite(osp.join(base_dir, 
#         str(label_train[i]), str(filename_train[i], 'utf-8')), img)

base_dir = osp.join(dir, 'test')
os.makedirs(base_dir, exist_ok=True)
for i in tqdm(range(data_test.shape[0])):
    if not osp.exists(osp.join(base_dir, str(label_test[i]))):
        os.makedirs(osp.join(base_dir, str(label_test[i])))
    r = data_test[i][:1024].reshape(32, 32)
    g = data_test[i][1024:2048].reshape(32, 32)
    b = data_test[i][2048:].reshape(32, 32)
    img = np.stack([b, g, r]).transpose((1, 2, 0))
    cv2.imwrite(osp.join(base_dir, 
        str(label_test[i]), str(filename_test[i], 'utf-8')), img)