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

from models.bilateral_gradcam import BilateralGradCAM
from models.macro import unilateral, bilateral, load_model, load_bilateral_model

from utils import run_cli

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def grad_cam(rgb_img, input_tensor, model, target_layers, use_cuda=True,   
                method_name="gradcam"):
    methods = {"gradcam": GradCAM, "bilateralgradcam": BilateralGradCAM}
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

def unilateral_single_head_gradcam(args):
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    with open(args.file, 'r') as f:
        images = f.read().splitlines()

    config = run_cli(config_path=args.config)

    # must use 'f' prefix for fine head, because unilateral can have 2 heads, and if there is only one, use fine
    mydict = {
                "mode_out": config["hparams"].get("mode_out"),
                "mode_heads": config["hparams"].get("mode_heads"),
                "farch": config["hparams"].get("farch"),
                "fmodel_path": None,
                "ffreeze_params": True,
                "fine_k": config["hparams"].get("fine_k"),
                "fine_per_k": config["hparams"].get("fine_per_k"),
                "dropout": config["hparams"].get("dropout", 0.0)
            }
    model_args = Namespace(**mydict)
    model = unilateral(model_args)

    load_model(model, args.checkpoint)

    model.to("cuda")

    res1 = model.hemisphere.conv[7]
    res2 = model.hemisphere.conv[16]
    target_layers = [res1[-1], res2[-1]]
    
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

def bilateral_gradcam(args):
    test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    with open(args.file, 'r') as f:
        images = f.read().splitlines()

    config = run_cli(config_path=args.config)

    mydict = {
            "mode_out": config["hparams"]["mode_out"],
            "mode_heads": config["hparams"]["mode_heads"],
            "farch": config["hparams"].get("farch", None),
            "carch": config["hparams"].get("carch", None),
            "fmodel_path": None,
            "cmodel_path": None,
            "ffreeze_params": True,
            "cfreeze_params": True,
            "fine_k": config["hparams"].get("fine_k"),
            "fine_per_k": config["hparams"].get("fine_per_k"),
            "coarse_k": config["hparams"].get("coarse_k"),
            "coarse_per_k": config["hparams"].get("coarse_per_k"),
            "dropout": config["hparams"].get("dropout", 0.0),
            }
    model_args = Namespace(**mydict)
    model = bilateral(model_args)

    model = load_bilateral_model(model, args.checkpoint)

    model.to("cuda")
    
    fine_res1 = model.fine_hemi.conv[7]
    fine_res2 = model.fine_hemi.conv[16]
    coarse_res1 = model.coarse_hemi.conv[7]
    coarse_res2 = model.coarse_hemi.conv[16]

    target_layers = [fine_res1[-1], fine_res2[-1], coarse_res1[-1], coarse_res2[-2]]

    dest_dir = f"./{args.src_dir}/{args.mode}"
    if Path(dest_dir).exists() and Path(dest_dir).is_dir():
        shutil.rmtree(dest_dir)
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    for image in tqdm(images):
        img = pil_loader(image)
        rgb_img = np.asarray(img)[:, :, ::-1] / 255.0
        img = test_transform(img)
        img = img.unsqueeze(0).to("cuda")
        heatmap_img = grad_cam(rgb_img, img, model, target_layers, use_cuda=True, method_name="bilateralgradcam")
        dest_img_path = str(Path(dest_dir).resolve() / image.split("/")[-1])
        cv2.imwrite(dest_img_path, heatmap_img)

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Test model on the CIFAR-100 narrow or broad dataset.")
    parser.add_argument("-c", "--config", type=str,
                        default="../config/config.yaml")
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

    config = run_cli(config_path=args.config)
    
    if ('farch' in config['hparams'] and config['hparams']['farch'] != 'resnet9') or ('carch' in config['hparams'] and config['hparams']['carch'] != 'resnet9'):
        print("Invalid architecture")
        sys.exit(0)

    if config['hparams']['mode_hemis'] == "bilateral":
        bilateral_gradcam(args)
    elif config['hparams']['mode_hemis'] == "unilateral":
        unilateral_single_head_gradcam(args)
    else:
        print("Invalid mode_hemis")


# Run Command
# python grad_cam.py -c runs/unilateral_specialize-unilateral-resnet9-fine/20230524164403-seed0/config.yaml -ckpt runs/unilateral_specialize-unilateral-resnet9-fine/20230524164403-seed0/checkpoints/epoch=165-val_acc=0.689.ckpt --gpu -f analysis/grad_cam_images/distribution_images_nfbf-bicamntbt.txt

