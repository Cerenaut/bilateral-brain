from typing import List, Any, Tuple, Optional, Callable


import glob
import os.path as osp

import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms.transforms import CenterCrop, Normalize, \
                        RandomErasing, RandomHorizontalFlip

import lightning as pl

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class HemiSphere(Dataset):
    def __init__(
            self,
            root: str,
            raw_data_path: str,
            transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            mode: Optional[str] = 'train',
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(HemiSphere, self).__init__()
        self.transform = transform
        self.loader = loader
        self.mode = mode
        valid_ext = IMG_EXTENSIONS if is_valid_file is None else None
        
        self.samples = []
        for ext in valid_ext:
            self.samples.extend(glob.glob(osp.join(root, '*', f'*{ext}')))
        
        self._load_coarse_labels(raw_data_path)

    def _load_coarse_labels(self, raw_data_path):
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        data_path = raw_data_path + self.mode
        data_dict = unpickle(data_path)
        del data_dict[b'data']
        coarse_labels = np.array(data_dict[b'coarse_labels'])
        filenames = data_dict[b'filenames']
        filenames = [x.decode("utf-8")for x in filenames]
        self.coarse_label_dict = dict(zip(filenames, coarse_labels))

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = int(path.split('/')[-2])
        coarselabel = self.coarse_label_dict[path.split('/')[-1]]
        # print("labels --------- " + str(target) + ', ' + str(coarselabel))
        data = {"image": sample, "fine": target, "coarse": coarselabel}
        return data
    
    def __len__(self):
        return len(self.samples)

class EnsembleHemiSphere(Dataset):
    def __init__(
            self,
            root: str,
            raw_data_path: str,
            transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            mode: Optional[str] = 'train',
            is_valid_file: Optional[Callable[[str], bool]] = None,
            split_file: Optional[str] = "/home/chandramouli/kaggle/cerenaut/analysis/train_split1.txt"
    ):
        super(EnsembleHemiSphere, self).__init__()
        self.transform = transform
        self.loader = loader
        self.mode = mode
        valid_ext = IMG_EXTENSIONS if is_valid_file is None else None

        images_fname  = None
        with open(split_file, "r") as f:
            images_fname = f.read().splitlines()
        self.images_fname = images_fname
        self.samples = []
        for ext in valid_ext:
            self.samples.extend(glob.glob(osp.join(root, '*', f'*{ext}')))
        self.samples = list(set(self.samples).intersection(set(self.images_fname)))
        self._load_coarse_labels()

    def _load_coarse_labels(self, raw_data_path):
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        data_path = raw_data_path + self.mode
        data_dict = unpickle(data_path)
        del data_dict[b'data']
        coarse_labels = np.array(data_dict[b'coarse_labels'])
        filenames = data_dict[b'filenames']
        filenames = [x.decode("utf-8")for x in filenames]
        self.coarse_label_dict = dict(zip(filenames, coarse_labels))

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = int(path.split('/')[-2])
        coarselabel = self.coarse_label_dict[path.split('/')[-1]]
        data = {"image": sample, "fine": target, "coarse": coarselabel}
        return data
    
    def __len__(self):
        return len(self.samples)


class DataModule(pl.LightningDataModule):
    def __init__(self, 
                    train_dir: str = "path/to/dir", 
                    val_dir: str="path/to/dir",
                    test_dir: str="path/to/dir",
                    raw_data_dir: str="path/to/dir",
                    batch_size: int = 32,
                    num_workers:int = 4,
                    split: bool = False,
                    split_file: str="/path/to/split"):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.raw_data_dir = raw_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_file = split_file
        self.split = split

    def setup(self, stage: Optional[str] = None):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
            transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        base_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        if self.split:
            self.mnist_train = EnsembleHemiSphere(
                root=self.train_dir,
                raw_data_path=self.raw_data_dir,
                transform=train_transforms,
                split_file=self.split_file)
        else:    
            self.mnist_train = HemiSphere(
              root=self.train_dir,
              raw_data_path=self.raw_data_dir,
              transform=train_transforms)        
        self.mnist_val = HemiSphere(
              root=self.val_dir,
              raw_data_path=self.raw_data_dir,
              mode='test',
              transform=base_transforms)
        self.mnist_test = HemiSphere(
              root=self.test_dir,
              raw_data_path=self.raw_data_dir,
              mode='test',
              transform=base_transforms)        

    def train_dataloader(self):
        return DataLoader(self.mnist_train,
                            shuffle=True,
                            pin_memory=True,
                            batch_size=self.batch_size, 
                            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, 
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test,
                            shuffle=False,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
    
if __name__ == '__main__':
    base_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])
    train_dir='/home/chandramouli/Documents/kaggle/CIFAR-100/val'
    dataset = EnsembleHemiSphere(
            root=train_dir,
            transform=base_transforms,
            split_file="/home/chandramouli/kaggle/cerenaut/analysis/val_split2.txt")
    d = DataLoader(dataset,
                            shuffle=True,
                            pin_memory=True,
                            batch_size=32, 
                            num_workers=4)