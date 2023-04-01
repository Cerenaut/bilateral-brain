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
from supervised import SupervisedLightningModule
from utils import run_cli, yaml_func


def main(config_path) -> None:
    config = run_cli(config_path=config_path)
    seeds = config['seeds']
    for seed in seeds:
        if seed is not None:
            pl.seed_everything(seed)

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

        logger = TensorBoardLogger(
            save_dir=config['logger']['save_dir'],
            name=config['logger']['name']+f"-seed{seed}",
            version=config['logger']['version'],)
        
        dest_dir = os.path.join(config['logger']['save_dir'], config['logger']['name']+f"-seed{seed}", f"{config['logger']['version']}")

        trainer = pl.Trainer(**config['trainer_params'],
                            callbacks=[ckpt_callback],
                            logger=logger)
        imdm = DataModule(
            train_dir=config['dataset']['train_dir'],
            val_dir=config['dataset']['val_dir'],
            batch_size=config['hparams']['batch_size'],
            num_workers=config['hparams']['num_workers'])
        trainer.fit(model, datamodule=imdm)
        shutil.copy(config_path, f'{dest_dir}/config.yaml')

if __name__ == '__main__':
  default_config_path = './configs/config.yaml'
  main(default_config_path)
