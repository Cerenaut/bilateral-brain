import os
import sys
import yaml
import optuna
import shutil
import os.path as osp
import pytorch_lightning as pl
from datetime import datetime

from pathlib import Path
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger, TensorBoardLogger

from datamodule import DataModule
from supervised import SupervisedLightningModule

import sys
sys.path.append('../')
from utils import run_cli, validate_path, yaml_func


def main(config_path) -> None:
    config = run_cli(config_path=config_path)
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
               
        model = SupervisedLightningModule(config)

        save_dir = config['save_dir']
        exp_name = f"{config['exp_name']}-{config['hparams']['farch']}-{config['hparams']['carch']}-{config['hparams']['mode']}"
        date_time = datetime.now().strftime("%Y%m%d%H%M%S")
        version = f"{date_time}-seed{seed}"
        
        logger = TensorBoardLogger(save_dir=save_dir, name=exp_name, version=version)

        trainer = pl.Trainer(**config['trainer_params'],
                            callbacks=[ckpt_callback],
                            logger=logger)

        imdm = DataModule(
            train_dir=config['dataset']['train_dir'],
            val_dir=config['dataset']['test_dir'],
            test_dir=config['dataset']['test_dir'],
            raw_data_dir=config['dataset']['raw_data_dir'],
            batch_size=config['hparams']['batch_size'],
            num_workers=config['hparams']['num_workers'],
            split=False,
            split_file=None)
        trainer.fit(model, datamodule=imdm)

        dest_dir = os.path.join(save_dir, exp_name, version)
        shutil.copy(config_path, f'{dest_dir}/config.yaml')

        if config["evaluate"]:
            acc = trainer.test(datamodule=imdm)
            
            # write the results to a file
            with open(f'{dest_dir}/results.txt', 'a') as f:
                f.write(f'{acc}\n')

if __name__ == '__main__':
    default_config_path = './configs/config.yaml'
    main(default_config_path)
