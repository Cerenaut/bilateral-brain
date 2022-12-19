from typing import List, Any, Tuple, Optional, Callable
from PIL import Image

import torch
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.transforms.transforms import CenterCrop, Normalize,            RandomErasing, RandomHorizontalFlip

import pytorch_lightning as pl

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


class DataModule(pl.LightningDataModule):
    def __init__(self, 
                    train_dir: str = "path/to/dir", 
                    val_dir: str="path/to/dir",
                    batch_size: int = 32):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = 8

    def setup(self, stage: Optional[str] = None):
        base_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])
        self.mnist_train = UnsupervisedFolder(
            root=self.train_dir,
            transform=base_transforms)
        
        self.mnist_val = UnsupervisedFolder(
            root=self.val_dir,
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
