import os
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm

''' 
This script reads from the cifar pickle 
and writes the images into appropriately named folders
for coarse and fine labels for both the train and test splits

Note that the same image is written to two locations, one for each label type 
because pytorch gets the name of the label from the folder
'''

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
# set paths here
# ------------------------------------------------------
metadata_path = '/Users/gideon/Dev/datasets/cifar-100-python/meta' # source cifar 'meta' file
data_pre_path = '/Users/gideon/Dev/datasets/cifar-100-python/' # source cifar files here (must have a trailing slash)
target_dir = 'datasets/CIFAR100'    # the output images go here
# ------------------------------------------------------

metadata = unpickle(metadata_path)
superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))

# file paths
data_train_path = data_pre_path + 'train'
data_test_path = data_pre_path + 'test'

# read dictionary
data_train_dict = unpickle(data_train_path)
data_test_dict = unpickle(data_test_path)

# prepare first level of the loop
base_dir_dic = {'train': osp.join(target_dir, 'train'), 'test': osp.join(target_dir, 'test')}
filenames_dic = {'train': data_train_dict[b'filenames'], 'test': data_test_dict[b'filenames']}
data_dic = {'train': data_train_dict[b'data'], 'test': data_test_dict[b'data']}

# prepare the second level of the loop
coarse_label_train = np.array(data_train_dict[b'coarse_labels'])
fine_label_train = np.array(data_train_dict[b'fine_labels'])
coarse_label_test = np.array(data_test_dict[b'coarse_labels'])
fine_label_test = np.array(data_test_dict[b'fine_labels'])

coarse_label_dic = {'train': coarse_label_train, 'test': coarse_label_test}
fine_label_dic = {'train': fine_label_train, 'test': fine_label_test}

# for: train, test
for split in ['train', 'test']:
  os.makedirs(base_dir_dic[split], exist_ok=True)

  label_dic_arr = [coarse_label_dic, fine_label_dic]
  label_type_arr = ['coarse', 'fine']
  
  print('----> writing images for {}'.format(split))
  for i in tqdm(range(data_dic[split].shape[0])):
    
    r = data_dic[split][i][:1024].reshape(32, 32)
    g = data_dic[split][i][1024:2048].reshape(32, 32)
    b = data_dic[split][i][2048:].reshape(32, 32)
    img = np.stack([b, g, r]).transpose((1, 2, 0))

    # for: coarse, fine
    for label_dic, label_type in zip(label_dic_arr, label_type_arr):
      out_dir = osp.join(base_dir_dic[split], label_type)
      if not osp.exists(osp.join(out_dir, str(label_dic[split][i]))):
          os.makedirs(osp.join(out_dir, str(label_dic[split][i])))

      cv2.imwrite(osp.join(out_dir, 
          str(label_dic[split][i]), str(filenames_dic[split][i], 'utf-8')), img)