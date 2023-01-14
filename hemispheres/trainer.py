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

from utils import run_cli, validate_path, yaml_func
from datamodule import DataModule
from supervised import SupervisedLightningModule


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
            train_dir=config['dataset']['train_dir'],
            val_dir=config['dataset']['val_dir'],
            batch_size=config['hparams']['batch_size'],
            num_workers=config['hparams']['num_workers'],
            split=False,
            split_file=None)
        trainer.fit(model, datamodule=imdm)


if __name__ == '__main__':
    cli_lightning_logo()
    main()
