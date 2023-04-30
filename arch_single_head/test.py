from typing import List, Any, Tuple, Optional, Callable

import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from datamodule import UnsupervisedFolder

from model import SparseAutoencoder
from sklearn.metrics import accuracy_score
from utils import pil_loader, run_cli


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


config = run_cli()

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
    
test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])

dataset = UnsupervisedFolder(config['dataset']['test_dir'],
                        transform=test_transforms,
                        loader=pil_loader,)

dataloader = DataLoader(dataset,
                            drop_last=False,
                            batch_size=32, 
                            num_workers=4)

model = SparseAutoencoder(num_input_channels=3,
                            base_channel_size=32,
                            num_classes=100).to(device)
checkpoint = torch.load(config['hparams']['model_path'])
checkpoint['state_dict'] = {k.replace('model.',''):v \
                for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(checkpoint['state_dict'])
model.eval()
targets = []
outputs = []
for (ind, batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
    img, target = batch
    img = img.to(device)
    out = model(img)
    out = out.detach().cpu()
    targets.append(target.detach().cpu())
    outputs.append(out)

targets_end = []
outputs_end = []
for out in list(zip(outputs, targets)):
    labels, target = out
    _, ind = torch.max(labels, 1)
    targets_end.append(target)
    outputs_end.append(ind)
targets = torch.cat(targets_end, 0).numpy()
outputs = torch.cat(outputs_end, 0).numpy()
acc1 = accuracy_score(targets, outputs)
print(acc1)


