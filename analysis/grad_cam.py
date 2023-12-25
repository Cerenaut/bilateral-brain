from typing import List, Any, Tuple, Optional, Callable

import cv2
import sys
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from argparse import Namespace

import torch
from torchvision import models
from torchvision import transforms as T

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append("/kaggle/cerenaut/")
from analysis.bicameral_gradcam import BicameralGradCAM
from models.sparse_resnet import load_model, load_feat_model, \
                load_bicam_model,sparse_resnet9, bicameral
from models.resnet import resnet18, resnet34, resnet50
from models.vgg import vgg11

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def grad_cam(rgb_img, input_tensor, model, target_layers, use_cuda=True,   
                method_name="gradcam"):
    methods = {"gradcam": GradCAM, "bicameralgradcam": BicameralGradCAM}
    cam_algorithm = methods[method_name]
    cam = cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=use_cuda)
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None)[0, :]
    # cv2.imshow("grayscale", grayscale_cam)
    # cv2.waitKey(0)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
    return visualization

def baseline_images(args):
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    with open(args.file, 'r') as f:
        images = f.read().splitlines()
    if "resnet50" in args.mode:
        model = resnet50()
        target_layers = [model.conv4_x[-1], model.conv5_x[-1]]
    elif "resnet34" in args.mode:
        model = resnet34()
        target_layers = [model.conv4_x[-1], model.conv5_x[-1]]
    elif "vgg11" in args.mode:
        model = vgg11()
        target_layers = [model.features[-2], model.features[-3]]
    else:
        model = resnet18()
        target_layers = [model.conv4_x[-1], model.conv5_x[-1]]
    model = load_model(model, args.checkpoint)
    model.to("cuda")
    # model.eval()
    
    dest_dir = f"./{args.src_dir}/{args.mode}"
    if Path(dest_dir).exists() and Path(dest_dir).is_dir():
        shutil.rmtree(dest_dir)
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    for image in tqdm(images):
        img = pil_loader(image)
        rgb_img = np.asarray(img)[:, :, ::-1] / 255.0
        img = test_transform(img)
        img = img.unsqueeze(0).to("cuda")
        heatmap_img = grad_cam(rgb_img, img, model, target_layers, use_cuda=True, method_name="gradcam")
        dest_img_path = str(Path(dest_dir).resolve() / image.split("/")[-1])
        cv2.imwrite(dest_img_path, heatmap_img)

def specialised_images(args):
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    with open(args.file, 'r') as f:
        images = f.read().splitlines()
    mydict = {
                "mode": args.mode,
                "k": None,
                "k_percent": None,
        }
    model_args = Namespace(**mydict)
    model = sparse_resnet9(model_args)
    model = load_model(model, args.checkpoint)
    model.to("cuda")
    # model.eval()
    target_layers = [model.res1[-1], model.res2[-1]]
    dest_dir = f"./{args.src_dir}/{args.mode}"
    if Path(dest_dir).exists() and Path(dest_dir).is_dir():
        shutil.rmtree(dest_dir)
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    for image in tqdm(images):
        img = pil_loader(image)
        rgb_img = np.asarray(img)[:, :, ::-1] / 255.0
        img = test_transform(img)
        img = img.unsqueeze(0).to("cuda")
        heatmap_img = grad_cam(rgb_img, img, model, target_layers, use_cuda=True, method_name="gradcam")
        dest_img_path = str(Path(dest_dir).resolve() / image.split("/")[-1])
        cv2.imwrite(dest_img_path, heatmap_img)

def bicameral_images(args):
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    with open(args.file, 'r') as f:
        images = f.read().splitlines()
    mydict = {
                "carch": "resnet9",
                "farch": "resnet9",
                "mode": "feature",
                "cmodel_path": None,
                "fmodel_path": None,
                "ffreeze_params": True,
                "cfreeze_params": True,
                "bicam_mode": "both"
        }
    model_args = Namespace(**mydict)
    model = bicameral(model_args)
    model = load_bicam_model(model, args.checkpoint)
    model.to("cuda")
    target_layers = [model.broad.res1[-1], model.broad.res2[-1], model.narrow.res1[-1], model.narrow.res2[-1]]
    dest_dir = f"./{args.src_dir}/{args.mode}"
    if Path(dest_dir).exists() and Path(dest_dir).is_dir():
        shutil.rmtree(dest_dir)
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    for image in tqdm(images):
        img = pil_loader(image)
        rgb_img = np.asarray(img)[:, :, ::-1] / 255.0
        img = test_transform(img)
        img = img.unsqueeze(0).to("cuda")
        heatmap_img = grad_cam(rgb_img, img, model, target_layers, use_cuda=True, method_name="bicameralgradcam")
        dest_img_path = str(Path(dest_dir).resolve() / image.split("/")[-1])
        cv2.imwrite(dest_img_path, heatmap_img)

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Test model on the CIFAR-100 narrow or broad dataset.")
    parser.add_argument("-m", "--mode", default="broad", 
                        choices=["broad", "narrow", "bicameral", "resnet18", "resnet34", "resnet50", "vgg11"])
    parser.add_argument("-ckpt", "--checkpoint", type=str,
                        help="")
    parser.add_argument("-b", "--batch-size", type=int, default=4,
                        help="")
    parser.add_argument("-f", "--file", type=str, 
                        default="",
                        help="")
    parser.add_argument("--gpu",
                        action="store_true", )
    args = parser.parse_args()
    args.gpu = args.gpu and torch.cuda.is_available()
    if args.gpu:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    filename = args.file
    filename = filename.split("_")[-1].replace(".txt", "")
    args.src_dir = f"grad_cam_{filename}"
    if args.mode == "bicameral":
        bicameral_images(args)
    elif "broad" in args.mode or "narrow" in args.mode:
        specialised_images(args)
    else:
        baseline_images(args)

# Run Command
# Broad - python grad_cam.py -m broad -ckpt "/home/chandramouli/kaggle/cerenaut/classification/logs/broad/left-right-brain-broad-class/layer=1|lr=0.0001|wd=1.0e-5|bs=32|opt=adam|/checkpoints/epoch=93-val_acc=0.75.ckpt" --gpu -f ./distribution_images.txt
# Narrow - python grad_cam.py -m narrow -ckpt "/home/chandramouli/kaggle/cerenaut/classification/logs/narrow/left-right-brain-narrow-class/layer=1|lr=0.01|wd=1.0e-5|bs=32|opt=adam|/checkpoints/epoch=93-val_acc=0.66.ckpt" --gpu -f ./distribution_images.txt
# Bicameral - python grad_cam.py -m bicameral -ckpt "/home/chandramouli/kaggle/cerenaut/hemispheres/logs/bicameral_specialised/hemisphere/layer=only1|lr=0.0001|dropout=0.6|/checkpoints/epoch=29-val_loss=1.73.ckpt" --gpu -f ./distribution_images.txt