import os
import sys
import yaml
import optuna
import os.path as osp
import pytorch_lightning as pl

from pathlib import Path
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger, TensorBoardLogger

from datamodule import DataModule
from supervised import SupervisedLightningModule

import sys
sys.path.append('../')
from utils import run_cli, validate_path, yaml_func


def main() -> None:
    config = run_cli()
    seeds = config['seeds']
    for seed in seeds:
        if seed is not None:
            pl.seed_everything(seed)


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

        logger = TensorBoardLogger(
            config['logger']['save_dir'],
            name=config['logger']['name'] + f"-seed{seed}",
            version=config['logger']['version'],)

        trainer = pl.Trainer(**config['trainer_params'],
                            callbacks=[ckpt_callback],
                            logger=logger)
        imdm = DataModule(
            train_dir=config['dataset']['train_dir'],
            val_dir=config['dataset']['test_dir'],
            raw_data_dir=config['dataset']['raw_data_dir'],
            batch_size=config['hparams']['batch_size'],
            num_workers=config['hparams']['num_workers'],
            split=False,
            split_file=None)
        trainer.fit(model, datamodule=imdm)


if __name__ == '__main__':
    main()
