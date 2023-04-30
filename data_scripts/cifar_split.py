import os
import glob
import shutil
import random
import os.path as osp
from tqdm import tqdm
base_folder = '/home/chandramouli/Documents/kaggle/CIFAR-100-Coarse'
number_of_dirs = len(glob.glob(osp.join(base_folder, 'train/*/')))
# for i in tqdm(range(10, 20)):
#     total_files = glob.glob(osp.join(base_folder, f'train/{i}', '*.png'))
#     val_files = random.sample(total_files, int(len(total_files) * 0.2))
#     train_files = list(set(total_files) - set(val_files))
#     os.makedirs(osp.join(base_folder, f'val/{i}'), exist_ok=True)
#     for j in val_files:
#         image_name = j.split('/')[-1]
#         shutil.move(j, osp.join(base_folder, f'val/{i}/{image_name}'))