from typing import List, Any, Tuple, Optional, Callable


import glob
import os.path as osp

import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.transforms import CenterCrop, Normalize, \
                        RandomErasing, RandomHorizontalFlip
from torchvision.datasets import DatasetFolder
import os.path

import lightning as pl
from utils import setup_logger

logger = setup_logger(__name__)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')

TRAIN_VAL_SPLIT = 0.0   # proportion of training set to use for train, the remainder for validation

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class SingleHeadDataset(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(SingleHeadDataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_1 = self.transform(sample)

        return sample_1, target

    # def __len__(self):
    #     return 100


class DualHeadsDataset(Dataset):
    def __init__(
            self,
            root: str,
            mode: str,
            raw_data_path: str,
            transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(DualHeadsDataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.loader = loader
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
        # logging.debug("labels --------- " + str(target) + ', ' + str(coarselabel))
        data = {"image": sample, "fine": target, "coarse": coarselabel}
        return data
    
    def __len__(self):
        return len(self.samples)

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 mode_heads: str = 'both',
                 train_dir: str = "path/to/dir", 
                 test_dir: str="path/to/dir",
                 raw_data_dir: str="path/to/dir",
                 batch_size: int = 32,
                 num_workers:int = 4):
        super().__init__()
        self.train_dir = self.get_data_dir(mode_heads, train_dir)
        self.test_dir = self.get_data_dir(mode_heads, test_dir)
        self.raw_data_dir = raw_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode_heads = mode_heads

    def get_data_dir(self, mode_heads, base_folder):
        """
        Returns the path to the data directory

        If there are two heads (fine and coarse), then get the 'fine' labels from the image file
        and get the coarse labelsl from the raw data file

        If there is only one head, then get the labels from the image file
        """
        if mode_heads == 'fine' or mode_heads == 'both':
            suffix = 'fine'
        elif mode_heads == 'coarse':
            suffix = 'coarse'
        else:
            raise ValueError(f"mode_out {mode_heads} not recognized")

        folder_name = os.path.join(base_folder, suffix)

        return folder_name

    def setup_data_dual_heads(self, train_transforms, base_transforms):
        logger.debug("DataModule - setup double heads")

        train_set = DualHeadsDataset(
            mode='train',
            root=self.train_dir,
            raw_data_path=self.raw_data_dir,
            transform=train_transforms)
        
        # special case of no split, then use test set for validation
        if TRAIN_VAL_SPLIT == 0.0 or TRAIN_VAL_SPLIT == None:
            logger.debug("Using test set for validation")

            val_set = DualHeadsDataset(
                mode='test',
                root=self.test_dir,
                raw_data_path=self.raw_data_dir,
                transform=base_transforms)
            
            self.val_set = val_set
            self.train_set = train_set
        else:
            logger.debug(f"Using {TRAIN_VAL_SPLIT} proportion of train set for train, remainder for validation")
            self.train_set, self.val_set = self.get_train_val_splits(train_set)  

        self.test_set = DualHeadsDataset(
            mode='test',
            root=self.test_dir,
            raw_data_path=self.raw_data_dir,
            transform=base_transforms)
        
    def setup_data_single_head(self, train_transforms, base_transforms):
        logger.debug("DataModule - setup single head")

        train_set = SingleHeadDataset(
            root=self.train_dir,
            transform=train_transforms)

        # special case of no split, then use test set for validation
        if TRAIN_VAL_SPLIT == 0.0 or TRAIN_VAL_SPLIT == None:
            logger.debug("Using test set for validation")

            val_set = SingleHeadDataset(
                root=self.test_dir,
                transform=base_transforms)
            
            self.val_set = val_set
            self.train_set = train_set
        else:
            logger.debug(f"Using {TRAIN_VAL_SPLIT} proportion of train set for train, remainder for validation")
            self.train_set, self.val_set = self.get_train_val_splits(train_set)            

        self.test_set = SingleHeadDataset(
            root=self.test_dir,
            transform=base_transforms)
            
    def get_train_val_splits(self, train_set):
        train_set_size = int(len(train_set) * TRAIN_VAL_SPLIT)
        valid_set_size = len(train_set) - train_set_size
        train_set, val_set = random_split(train_set, [train_set_size, valid_set_size])
        return train_set, val_set

    def setup(self, stage: Optional[str] = None):
        logger.debug("*********** DataModule - setup ***********")
        train_transform = transforms.Compose([
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
        
        if self.mode_heads == 'both':
            self.setup_data_dual_heads(train_transform, base_transforms)
        else:
            self.setup_data_single_head(train_transform, base_transforms)

        logger.debug(f"Train set size: {len(self.train_set)}")
        # logger.debug(f"Train set transformation: {self.train_set.transform}")

        logger.debug(f"Validation set size: {len(self.val_set)}")
        # logger.debug(f"Validation set transformation: {self.val_set.transform}")

        logger.debug(f"Test set size: {len(self.test_set)}")
        # logger.debug(f"Test set transformation: {self.test_set.transform}")

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          shuffle=True,
                            batch_size=self.batch_size, 
                            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
    