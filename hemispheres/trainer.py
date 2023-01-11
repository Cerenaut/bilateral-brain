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
    seeds = [0, 20, 42, 80, 100]
    for seed in seeds:
        config['seed'] = seed
        if config['seed'] is not None:
            pl.seed_everything(config['seed'])

        ckpt_callback = ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}',
            **config['ckpt_callback'],
        )
        if 'callbacks' in config['trainer_params']:
            config['trainer_params']['callbacks'] = yaml_func(
                config['trainer_params']['callbacks'])
        if config['trainer_params']['default_root_dir'] == "None":
            config['trainer_params']['default_root_dir'] = osp.dirname(__file__)
        
        lr = config['hparams']['lr']
        
        config['logger']['version'] =f'layer=only1|lr={lr}|'
        
        model = SupervisedLightningModule(config)

        logger = TestTubeLogger(
            config['logger']['save_dir'],
            name=config['logger']['name'] + f"-seed{config['seed']}",
            version=config['logger']['version'],)
        trainer = pl.Trainer(**config['trainer_params'],
                            callbacks=[ckpt_callback],
                            logger=logger,
                            )
        imdm = DataModule(
            train_dir='/path/to/train_dataset',
            val_dir='/path/to/test_dataset',
            batch_size=config['hparams']['batch_size'],
            num_workers=config['hparams']['num_workers'],
            split=False,
            split_file=None)
        trainer.fit(model, datamodule=imdm)


if __name__ == '__main__':
    cli_lightning_logo()
    main()
