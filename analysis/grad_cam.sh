#!/bin/sh
python grad_cam.py -m broad -ckpt "/home/chandramouli/kaggle/cerenaut/classification/logs/broad/left-right-brain-broad-class/layer=1|lr=0.0001|wd=1.0e-5|bs=32|opt=adam|/checkpoints/epoch=93-val_acc=0.75.ckpt" --gpu -f $1
python grad_cam.py -m narrow -ckpt "/home/chandramouli/kaggle/cerenaut/classification/logs/narrow/left-right-brain-narrow-class/layer=1|lr=0.01|wd=1.0e-5|bs=32|opt=adam|/checkpoints/epoch=93-val_acc=0.66.ckpt" --gpu -f $1
python grad_cam.py -m bicameral -ckpt "/home/chandramouli/kaggle/cerenaut/hemispheres/logs/bicameral_specialised/hemisphere/layer=only1|lr=0.0001|dropout=0.6|/checkpoints/epoch=29-val_loss=1.73.ckpt" --gpu -f $1
python grad_cam.py -m resnet18 -ckpt "" --gpu -f $1
python grad_cam.py -m resnet34 -ckpt "" --gpu -f $1
python grad_cam.py -m resnet50 -ckpt "" --gpu -f $1
python grad_cam.py -m vgg11 -ckpt "" --gpu -f $1