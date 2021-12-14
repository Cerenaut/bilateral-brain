import os
import sys
import yaml
import optuna
import os.path as osp
import pytorch_lightning as pl

from pathlib import Path
from pl_examples import cli_lightning_logo
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger, TensorBoardLogger

from utils import validate_path
from datamodule import DataModule
from supervised import SupervisedLightningModule


def yaml_func(config_param):

    if isinstance(config_param, list):
        call_list = []
        local_func = locals().keys()
        for param in config_param:
            if param in local_func:
                call_list.append(locals()[param])
        return call_list
    elif isinstance(config_param, dict):
        call = None
        global_func = globals().keys()
        key = config_param['type']
        del config_param['type']
        if key in global_func:
            call = globals()[key](**config_param)
        return call

def run_cli():
    validate_path('./config.yaml')
    with open('./config.yaml', 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def main() -> None:
    config = run_cli()
    if config['seed'] is not None:
        pl.seed_everything(config['seed'])

    ckpt_callback = ModelCheckpoint(
        filename='{epoch}-{val_acc:.2f}',
        **config['ckpt_callback'],
    )
    if 'callbacks' in config['trainer_params']:
        config['trainer_params']['callbacks'] = yaml_func(
            config['trainer_params']['callbacks'])
    if config['trainer_params']['default_root_dir'] == "None":
        config['trainer_params']['default_root_dir'] = osp.dirname(__file__)
    
    model = SupervisedLightningModule(config)

    logger = TestTubeLogger(
        config['logger']['save_dir'],
        name=config['logger']['name'],
        version=config['logger']['version'],)
    trainer = pl.Trainer(**config['trainer_params'],
                         callbacks=[ckpt_callback],
                         logger=logger,
                         )
    imdm = DataModule(
        train_dir='/data/CIFAR-100-Coarse/train',
        val_dir='/data/CIFAR-100-Coarse/test',
        batch_size=config['hparams']['batch_size'])
    trainer.fit(model, datamodule=imdm)


if __name__ == '__main__':
    cli_lightning_logo()
    main()
