from sparsity_for_both_general_and_specific import generate_config as generate_config
from sparsity_for_both_general_and_specific import check_config as check_config

import os
import sys
import optuna
import shutil
import os.path as osp
import pytorch_lightning as pl

from pathlib import Path
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import DataModule
from supervised1 import SupervisedLightningModule

import sys
sys.path.append('../')
from utils import run_cli, yaml_func



from trainer import run_cli as run_cli
from trainer import main as main


#tem_alist=list(range(0,11,1))
#blist=[x/10 for x in tem_alist]
#print(blist)
#blist=[0.25,0.5,0.75]

class111=["coarse","fine"]
k111=[0.01]
perk111=[1]
resnet_or_not=[0]

class_k_per_k=[]


for a in range(len(class111)):
    for b in range(len(k111)):
        for c in range(len(perk111)):
            for d in range(len(resnet_or_not)):
                config_name=generate_config(class111[a],k111[b],perk111[c],resnet_or_not[d])
                check_config(config_name)
                class_k_per_k.append(config_name)




for t in range(len(class_k_per_k)):
    main(class_k_per_k[t])
