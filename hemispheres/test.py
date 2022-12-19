from typing import List, Any, Tuple, Optional, Callable

import glob
import torch
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from model import SparseAutoencoder, Combiner
from sklearn.metrics import accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        self._load_coarse_labels()

    def _load_coarse_labels(self):
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        data_pre_path = '/home/chandramouli/Downloads/cifar-100-python/'
        data_path = data_pre_path + self.mode
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
    
test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])
# 1.0e-4 acc = 2.375%
# 1.0e-3 acc = 1.875%

TEST_FOLDER = '/home/chandramouli/Documents/kaggle/CIFAR-100/test'
dataset = HemiSphere(root=TEST_FOLDER,
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

#Narrow Class Model
CKPT_PATH = '/home/chandramouli/kaggle/cerenaut/hemispheres/logs/hemisphere-non-pretrained/layer=only1|lr=0.0001|wd=1.0e-5|bs=32|opt=adam|/checkpoints/last.ckpt'
checkpoint = torch.load(CKPT_PATH)
checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items()\
     if "model_narrow" in k}
checkpoint['state_dict'] = {k.replace('model_narrow.',''):v \
                for k,v in checkpoint['state_dict'].items()}
model_narrow.load_state_dict(checkpoint['state_dict'])
model_narrow.eval()

#Broad Class Model
CKPT_PATH = '/home/chandramouli/kaggle/cerenaut/hemispheres/logs/hemisphere-non-pretrained/layer=only1|lr=0.0001|wd=1.0e-5|bs=32|opt=adam|/checkpoints/last.ckpt'
checkpoint = torch.load(CKPT_PATH)
checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items()\
     if "model_broad" in k}
checkpoint['state_dict'] = {k.replace('model_broad.',''):v \
                for k,v in checkpoint['state_dict'].items()}
model_broad.load_state_dict(checkpoint['state_dict'])
model_broad.eval()

#Combiner Model
CKPT_PATH = '/home/chandramouli/kaggle/cerenaut/hemispheres/logs/hemisphere-non-pretrained/layer=only1|lr=0.0001|wd=1.0e-5|bs=32|opt=adam|/checkpoints/last.ckpt'
checkpoint = torch.load(CKPT_PATH)
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

# Narrow accuracy 0.2351
# Broad accuracy 0.3267

# Narrow accuracy 0.224
# Broad accuracy 0.3193