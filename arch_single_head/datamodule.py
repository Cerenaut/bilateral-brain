from typing import List, Any, Tuple, Optional, Callable
from PIL import Image

import torch
import torchvision
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import lightning as pl

import sys
sys.path.append('../')
from utils import setup_logger

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class UnsupervisedFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(UnsupervisedFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
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

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                    train_dir: str = "path/to/dir", 
                    val_dir: str="path/to/dir",
                    test_dir: str="path/to/dir",
                    batch_size: int = 32,
                    num_workers: int = 1):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.logger = setup_logger()

    def setup(self, stage: Optional[str] = None):
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
        self.data_train = UnsupervisedFolder(
            root=self.train_dir,
            transform=train_transform)
        self.data_val = UnsupervisedFolder(
            root=self.val_dir,
            transform=base_transforms)
        self.data_test = UnsupervisedFolder(
            root=self.test_dir,
            transform=base_transforms)
        
    def train_dataloader(self):
        # self.logger.debug(f"Train dataset size: {len(self.data_train)}")
        return DataLoader(self.data_train, 
                            shuffle=True,
                            batch_size=self.batch_size, 
                            num_workers=self.num_workers)

    def val_dataloader(self):
        # self.logger.debug(f"Val dataset size: {len(self.data_val)}")
        return DataLoader(self.data_val,
                            shuffle=False,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)

    def test_dataloader(self):
        # self.logger.debug(f"Test dataset size: {len(self.data_test)}")
        return DataLoader(self.data_test,
                            shuffle=False,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)