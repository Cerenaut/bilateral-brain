from typing import List, Any, Tuple, Optional, Callable

import glob
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from arch_dual_head.datamodule import HemiSphere
from model import SparseAutoencoder, Combiner
from sklearn.metrics import accuracy_score
from utils import pil_loader, run_cli


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


config = run_cli()


test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])

dataset = HemiSphere(root=config['dataset']['test'],
                        transform=test_transforms,
                        loader=pil_loader,
                        mode='test',)

dataloader = DataLoader(dataset,
                            drop_last=False,
                            batch_size=32, 
                            num_workers=4)

model_narrow = SparseAutoencoder(num_input_channels=3,
                                        base_channel_size=32, 
                                        latent_dim=512,
                                        num_classes=100).to(device)
model_broad = SparseAutoencoder(num_input_channels=3,
                                base_channel_size=32, 
                                latent_dim=512,
                                num_classes=20).to(device)
combiner = Combiner().to(device)

# Fine Class Model
checkpoint = torch.load(config['hparams']['model_path_fine'])
checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items()\
     if "model_narrow" in k}
checkpoint['state_dict'] = {k.replace('model_narrow.',''):v \
                for k,v in checkpoint['state_dict'].items()}
model_narrow.load_state_dict(checkpoint['state_dict'])
model_narrow.eval()

# Broad Class Model
checkpoint = torch.load(config['hparams']['model_path_broad'])
checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items()\
     if "model_broad" in k}
checkpoint['state_dict'] = {k.replace('model_broad.',''):v \
                for k,v in checkpoint['state_dict'].items()}
model_broad.load_state_dict(checkpoint['state_dict'])
model_broad.eval()

# Combiner Model
checkpoint = torch.load(config['hparams']['model_path_combiner'])
checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items()\
     if "combiner" in k}
checkpoint['state_dict'] = {k.replace('combiner.',''):v \
                for k,v in checkpoint['state_dict'].items()}
combiner.load_state_dict(checkpoint['state_dict'])
combiner.eval()

fines = []
coarses = []
narrows = []
broads = []
for (ind, batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
    img, fine, coarse = batch['image'], batch['fine'], batch['coarse']
    img = img.to(device)
    enc_narrow, dec_narrow = model_narrow(img)
    enc_broad, dec_broad = model_broad(img)
    enc = torch.cat([enc_narrow, enc_broad], axis=-1)
    narrow, broad = combiner(enc)
    narrow = narrow.detach().cpu()
    broad = broad.detach().cpu()
    narrows.append(narrow)
    broads.append(broad)
    fines.append(fine)
    coarses.append(coarse)

coarses_end = []
fines_end = []
narrows_end = []
broads_end = []
for out in list(zip(narrows, fines, broads, coarses)):
    narrows, fines, broads, coarses = out
    _, narrow_ind = torch.max(narrows, 1)
    _, broad_ind = torch.max(broads, 1)
    broads_end.append(broad_ind)
    narrows_end.append(narrow_ind)
    coarses_end.append(coarses)
    fines_end.append(fines)
fines = torch.cat(fines_end, 0).numpy()
coarses = torch.cat(coarses_end, 0).numpy()
narrows = torch.cat(narrows_end, 0).numpy()
broads = torch.cat(broads_end, 0).numpy()
acc_narr = accuracy_score(fines, narrows)
acc_broad = accuracy_score(coarses, broads)
print(f"Narrow accuracy {acc_narr}")
print(f"Broad accuracy {acc_broad}")
