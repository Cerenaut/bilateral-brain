import os
import sys
import yaml
import shutil
import os.path as osp
import lightning as pl
from datetime import datetime
import numpy as np
import argparse

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


if __name__ == '__main__':
    from datamodule import DataModule
    from supervised import SupervisedLightningModule
else:
    from .datamodule import DataModule
    from .supervised import SupervisedLightningModule

from utils import run_cli, yaml_func


def main(config_path) -> None:

    config = run_cli(config_path=config_path)
    seeds = config['seeds']
    runs = []               # one for each seed
    accuracies_fine = []    # one for each seed
    accuracies_coarse = []  # one for each seed
    for seed in seeds:
        print(f"Running seed {seed}")

        if seed is not None:
            pl.seed_everything(seed)

        monitor_var = config['ckpt_callback']['monitor']
        ckpt_callback = ModelCheckpoint(
            filename='{epoch}-{' + monitor_var + ':.3f}',
            **config['ckpt_callback'],
        )

        if 'callbacks' in config['trainer_params']:
            config['trainer_params']['callbacks'] = yaml_func(
                config['trainer_params']['callbacks'])
        if config['trainer_params']['default_root_dir'] == "None":
            config['trainer_params']['default_root_dir'] = osp.dirname(__file__)
               
        model = SupervisedLightningModule(config)

        save_dir = config['save_dir']
        arch_string = ''
        if config['hparams']['macro_arch'] == 'ensemble':
            model_path_list = config['hparams']['model_path_fine']
            arch_string = f"{len(model_path_list)}x{config['hparams']['farch']}"
        else:
            arch_string = f"{config['hparams']['farch']}-{config['hparams']['carch']}"  # default name

        exp_name = f"{config['exp_name']}-{config['hparams']['macro_arch']}-{arch_string}-{config['hparams']['mode']}"
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
        
        dest_dir = os.path.join(save_dir, exp_name, version)
        if config['hparams']['macro_arch'] != 'ensemble':
            trainer.fit(model, datamodule=imdm)
        else:
            # create dest_dir (because it won't be created by training)
            os.makedirs(dest_dir, exist_ok=True)

        shutil.copy(config_path, f'{dest_dir}/config.yaml')

        if config["evaluate"]:
            result = trainer.test(model, datamodule=imdm)

            rdict = {
                'seed': seed,
                'result': result
            }
            runs.append(rdict)
            
            acc_fine = result[0]['test_acc_fine']
            acc_coarse = result[0]['test_acc_coarse']
            accuracies_fine.append(acc_fine)
            accuracies_coarse.append(acc_coarse)

    if config["evaluate"]:
        accs_fine = np.array(accuracies_fine)
        accs_coarse = np.array(accuracies_coarse)
        
        results = {
            'summary': 
                {
                    'mean_fine': str(accs_fine.mean()),
                    'stddev_fine': str(accs_fine.std()),
                    'mean_coarse': str(accs_coarse.mean()),
                    'stddev_coarse': str(accs_coarse.std()), 
                },
            'runs': runs
        }

        # write the results list to a yaml file
        with open(f'{dest_dir}/results.yaml', 'w') as f:
            yaml.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual hemisphere training/testing')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                    help='Path to the base config file for training macro-arch with 2 heads. Relative to the folder where you ran this from.')
    args = parser.parse_args()
    main(args.config)
