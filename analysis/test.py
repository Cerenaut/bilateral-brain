from typing import List, Any, Tuple, Optional, Callable

import sys
import glob

import argparse
import numpy as np
import os.path as osp

from PIL import Image
from pathlib import Path
from argparse import Namespace
from rich.progress import track
from torchvision import transforms as T
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader

sys.path.append("/kaggle/cerenaut/")
from models.sparse_resnet import load_model, load_feat_model, \
                load_bicam_model,sparse_resnet9, bicameral

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
        data = {
                "image": sample_1, 
                "target": target,
                "path": path
                }
        return data

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
        data = {
                "image": sample, 
                "fine": target, 
                "coarse": coarselabel,
                "path": path
                }
        return data
    
    def __len__(self):
        return len(self.samples)

def specialised_images(args):
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    with open(args.file, 'r') as f:
        images = f.read().splitlines()
    filetype = args.file.split('/')[-1].split('_')[-1].split('.')[0]
    Path(f"./grad_cam_{filetype}").mkdir(exist_ok=True, parents=True)
    mydict = {
                "mode": "feature",
                "k": None,
                "k_percent": None,
        }
    model_args = Namespace(**mydict)
    model = sparse_resnet9(model_args)
    model = load_feat_model(model, args.checkpoint)
    model.to("cuda")
    model.eval()
    feats = []
    for image in images:
        img = pil_loader(image)
        img = test_transform(img)
        img = img.unsqueeze(0).to("cuda")
        feat = model(img)
        feat = feat.detach().cpu().numpy()
        feats.append(feat)
    feats = np.stack(feats)
    with open(f'./grad_cam_{filetype}/{args.mode}_feat.npy', 'wb') as f:
        np.save(f, feats)
    
def specialised(args):
    if args.mode == "broad":
        val_dir = "/home/chandramouli/Documents/kaggle/CIFAR-100-Coarse/test"
    else:
        val_dir = "/home/chandramouli/Documents/kaggle/CIFAR-100/test"
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    mnist_test = UnsupervisedFolder(
                root=val_dir,
                transform=test_transform)
    dataloader = DataLoader(mnist_test, 
                            shuffle=False,
                            batch_size=args.batch_size, 
                            num_workers=4)
    mydict = {
                "mode": args.mode,
                "k": None,
                "k_percent": None,
        }
    model_args = Namespace(**mydict)
    model = sparse_resnet9(model_args)
    model = load_model(model, args.checkpoint)
    model.to("cuda")
    model = model.eval()
    length, correct = 0, 0
    result =  np.array([])
    for batch in track(dataloader):
        image, target, path = batch['image'], \
                                        batch['target'], \
                                        batch['path']
        image = image.to("cuda")
        output = model(image)
        pred = output.max(1)[1].cpu()
        correct += (pred.eq(target)).sum()
        result_batch = np.stack([np.array(path).astype(str), target.numpy().astype(np.int32), pred.numpy().astype(np.int32)], axis=1)
        length += image.shape[0]
    if result.shape[0] == 0:
        result = result_batch
    else:
        result = np.vstack([result, result_batch])
    accuracy = correct / length
    print(f"{args.mode} Accuracy : {accuracy}")
    np.savetxt(f"./{args.mode}.txt", result, fmt="%s,%s,%s")
    

def bicameral_images(args):
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    with open(args.file, 'r') as f:
        images = f.read().splitlines()
    filetype = args.file.split('/')[-1].split('_')[-1].split('.')[0]
    Path(f"./grad_cam_{filetype}").mkdir(exist_ok=True, parents=True)
    mydict = {
                "carch": "resnet9",
                "farch": "resnet9",
                "mode": "feature",
                "cmodel_path": None,
                "fmodel_path": None,
                "ffreeze_params": True,
                "cfreeze_params": True,
                "bicam_mode": "feature"
        }
    model_args = Namespace(**mydict)
    model = bicameral(model_args)
    model = load_bicam_model(model, args.checkpoint)
    model.to("cuda")
    model = model.eval()
    feats = []
    for image in images:
        img = pil_loader(image)
        img = test_transform(img)
        img = img.unsqueeze(0).to("cuda")
        feat = model(img)
        feat = feat.detach().cpu().numpy()
        feats.append(feat)
    feats = np.stack(feats)
    with open(f'./grad_cam_{filetype}/{args.mode}_feat.npy', 'wb') as f:
        np.save(f, feats)

    
def bicameral_batch(args):
    val_dir = "/home/chandramouli/Documents/kaggle/CIFAR-100/test"
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    mnist_test = HemiSphere(
            root=val_dir,
            mode='test',
            transform=test_transform)
    dataloader = DataLoader(mnist_test, 
                            shuffle=False,
                            batch_size=args.batch_size, 
                            num_workers=4)
    mydict = {
                "carch": "resnet9",
                "farch": "resnet9",
                "mode": "feature",
                "cmodel_path": None,
                "fmodel_path": None,
                "ffreeze_params": True,
                "cfreeze_params": True,
                "bicam_model": "both"
        }
    model_args = Namespace(**mydict)
    model = bicameral(model_args)
    model = load_bicam_model(model, args.checkpoint)
    model.to("cuda")
    model = model.eval()
    length, ccorrect, fcorrect = 0, 0, 0
    fresult, cresult = np.array([]), np.array([])
    for batch in track(dataloader):
        image, finey, coarsey, path = batch['image'], \
                                        batch['fine'], \
                                        batch['coarse'], \
                                        batch['path']
        image = image.to("cuda:0")
        foutput, coutput = model(image)
        fpred = foutput.max(1)[1].cpu()
        cpred = coutput.max(1)[1].cpu()
        fcorrect += (fpred.eq(finey)).sum()
        ccorrect += (cpred.eq(coarsey)).sum()
        fresult_batch = np.stack([np.array(path).astype(str), finey.numpy().astype(np.int32), fpred.numpy().astype(np.int32)], axis=1)
        cresult_batch = np.stack([np.array(path).astype(str), coarsey.numpy().astype(np.int32), cpred.numpy().astype(np.int32)], axis=1)
        length += image.shape[0]
    if fresult.shape[0] == 0 and cresult.shape[0] == 0:
        fresult = fresult_batch
        cresult = cresult_batch
    else:
        cresult = np.vstack([cresult, cresult_batch])
        fresult = np.vstack([fresult, fresult_batch])
    faccuracy = fcorrect / length
    caccuracy = ccorrect / length
    print(f"Narrow Accuracy : {faccuracy}")
    print(f"Broad Accuracy : {caccuracy}")
    np.savetxt(f"./bicameral_narrow.txt", fresult, fmt="%s,%s,%s")
    np.savetxt(f"./bicameral_broad.txt", cresult, fmt="%s,%s,%s")

def ensemble_images(args):
    val_dir = "/home/chandramouli/Documents/kaggle/CIFAR-100/test"
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    mnist_test = HemiSphere(
            root=val_dir,
            mode='test',
            transform=test_transform)
    dataloader = DataLoader(mnist_test, 
                            shuffle=False,
                            batch_size=args.batch_size, 
                            num_workers=4)
    mydict = {
                "mode": "both",
                "k": None,
                "k_percent": None,
        }
    model_args = Namespace(**mydict)
    model_a = sparse_resnet9(model_args)
    model_a = load_model(model_a, "/home/chandramouli/kaggle/cerenaut/hemispheres/logs/resnet9-ensemble5/resnet9-ensemble5-seed80/layer=only1|lr=0.0001|/checkpoints/last.ckpt")
    model_a.to("cuda")
    model_a = model_a.eval()
    model_b = sparse_resnet9(model_args)
    model_b = load_model(model_b, "/home/chandramouli/kaggle/cerenaut/hemispheres/logs/resnet9-ensemble5/resnet9-ensemble5-seed100/layer=only1|lr=0.0001|/checkpoints/last.ckpt")
    model_b.to("cuda")
    model_b = model_b.eval()
    # models = []
    # for f in glob.glob("/home/chandramouli/kaggle/cerenaut/hemispheres/logs/resnet9-ensemble5/*/*/checkpoints/last.ckpt"):
    #     model = resnet9(model_args)
    #     model = load_model(model, f)
    #     model.to("cuda")
    #     model = model.eval()
    #     models.append(model)
    length, ccorrect, fcorrect = 0, 0, 0
    fresult, cresult = np.array([]), np.array([])
    for batch in track(dataloader):
        image, finey, coarsey, path = batch['image'], \
                                        batch['fine'], \
                                        batch['coarse'], \
                                        batch['path']
        image = image.to("cuda:0")
        # foutputs, coutputs = [], []
        # for i in range(5):
        #     fout, cout = models[i](image)
        #     foutputs.append(fout)
        #     coutputs.append(cout)
        # foutput = sum(foutputs) / 5
        # coutput = sum(coutputs) / 5
        foutput_a, coutput_a = model_a(image)
        foutput_b, coutput_b = model_b(image)
        foutput = 0.5 * foutput_a + 0.5 * foutput_b
        coutput = 0.5 * coutput_a + 0.5 * coutput_b
        fpred = foutput.max(1)[1].cpu()
        cpred = coutput.max(1)[1].cpu()
        fcorrect += (fpred.eq(finey)).sum()
        ccorrect += (cpred.eq(coarsey)).sum()
        fresult_batch = np.stack([np.array(path).astype(str), finey.numpy().astype(np.int32), fpred.numpy().astype(np.int32)], axis=1)
        cresult_batch = np.stack([np.array(path).astype(str), coarsey.numpy().astype(np.int32), cpred.numpy().astype(np.int32)], axis=1)
        length += image.shape[0]
    if fresult.shape[0] == 0 and cresult.shape[0] == 0:
        fresult = fresult_batch
        cresult = cresult_batch
    else:
        cresult = np.vstack([cresult, cresult_batch])
        fresult = np.vstack([fresult, fresult_batch])
    faccuracy = fcorrect / length
    caccuracy = ccorrect / length
    print(f"Narrow Accuracy : {faccuracy}")
    print(f"Broad Accuracy : {caccuracy}")

def main(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    if args.mode == "ensemble":
        ensemble_images(args)
    else:
        if args.file != "":
            if args.mode == "bicameral":
                bicameral_images(args)
            else:
                specialised_images(args)
        else:
            if args.mode == "bicameral":
                bicameral_batch(args)
            else:
                specialised(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model on the CIFAR-100 narrow or broad dataset.")
    parser.add_argument("-m", "--mode", default="broad", 
                        choices=["broad", "narrow", "bicameral", "ensemble"])
    parser.add_argument("-ckpt", "--checkpoint", type=str,
                        help="")
    parser.add_argument("-b", "--batch-size", type=int, default=4,
                        help="")
    parser.add_argument("-f", "--file", type=str, 
                        default="",
                        help="")
    args = parser.parse_args()
    main(args)

# Run Command
# Broad - python test.py -m broad -ckpt "/home/chandramouli/kaggle/cerenaut/classification/logs/broad/left-right-brain-broad-class/layer=1|lr=0.0001|wd=1.0e-5|bs=32|opt=adam|/checkpoints/epoch=93-val_acc=0.75.ckpt" -f ./distribution_images.txt
# Narrow - python test.py -m narrow -ckpt "/home/chandramouli/kaggle/cerenaut/classification/logs/narrow/left-right-brain-narrow-class/layer=1|lr=0.01|wd=1.0e-5|bs=32|opt=adam|/checkpoints/epoch=93-val_acc=0.66.ckpt" -f ./distribution_images.txt
# Bicameral - python test.py -m bicameral -ckpt "/home/chandramouli/kaggle/cerenaut/hemispheres/logs/bicameral_specialised/hemisphere/layer=only1|lr=0.0001|dropout=0.6|/checkpoints/epoch=29-val_loss=1.73.ckpt" -f ./distribution_images.txt 