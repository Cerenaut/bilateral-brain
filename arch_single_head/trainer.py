import os
import sys
import yaml
import optuna
import shutil
import os.path as osp
import pytorch_lightning as pl
from datetime import datetime
import numpy as np
import logging

from pathlib import Path
from argparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == '__main__':
    from datamodule import DataModule
    from supervised import SupervisedLightningModule
else:
    from .datamodule import DataModule
    from .supervised import SupervisedLightningModule
from utils import run_cli, yaml_func, setup_logger

logger = setup_logger(__name__)

def main(config_path, logger_name='arch_single_head') -> None:

    config = run_cli(config_path=config_path)
    seeds = config['seeds']

    # data structures to store results - one for each seed
    runs, accuracies,checkpoints = [], [], []

    for seed in seeds:
        if seed is not None:
            pl.seed_everything(seed)

        ckpt_callback = ModelCheckpoint(
            filename='{epoch}-{val_acc:.3f}',
            **config['ckpt_callback'],
        )
        if 'callbacks' in config['trainer_params']:
            config['trainer_params']['callbacks'] = yaml_func(
                config['trainer_params']['callbacks'])
        if config['trainer_params']['default_root_dir'] == "None":
            config['trainer_params']['default_root_dir'] = osp.dirname(__file__)
        
        model = SupervisedLightningModule(config)

        save_dir = config['save_dir']
        exp_name = f"{config['exp_name']}-{config['hparams']['arch']}-{config['hparams']['mode']}"
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
            batch_size=config['hparams']['batch_size'],
            num_workers=config['hparams']['num_workers'])
        trainer.fit(model, datamodule=imdm)

        # get the path to the best checkpoint saved by the callback
        checkpoint_path = ckpt_callback.best_model_path
        checkpoints.append(checkpoint_path)

        # save the config file in the results folder
        dest_dir = os.path.join(save_dir, exp_name, version)
        shutil.copy(config_path, f'{dest_dir}/config.yaml')

        if config["evaluate"]:
            result = trainer.test(datamodule=imdm)

            rdict = {
                'seed': seed,
                'result': result
            }
            runs.append(rdict)
            
            acc = result[0]['test_acc']
            accuracies.append(acc)

    if config["evaluate"]:
        accs = np.array(accuracies)
        mean = accs.mean()
        stddev = accs.std()

        results = {
            'summary': 
                {
                    'mean': str(mean),
                    'stddev': str(stddev),
                },
            'runs': runs
        }

        # write the results list to a yaml file
        with open(f'{dest_dir}/results.yaml', 'w') as f:
            yaml.dump(results, f)

    return checkpoints

if __name__ == '__main__':
  default_config_path = './configs/config.yaml'
  main(default_config_path)
