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

from datamodule import DataModule
from supervised_dual_head import SupervisedLightningModuleDualHead
from supervised_single_head import SupervisedLightningModuleSingleHead

from utils import run_cli, yaml_func


def get_exp_names(config, seed):
    mode_hemis = config['hparams']['mode_hemis']

    save_dir = config['save_dir']
    arch_string = ''
    if mode_hemis == 'ensemble':
        model_path_list = config['hparams']['model_path_fine']
        arch_string = f"{len(model_path_list)}x{config['hparams']['farch']}"
    elif mode_hemis == 'bilateral':
        arch_string = f"{config['hparams']['farch']}-{config['hparams']['carch']}"  # default name
    elif mode_hemis == 'unilateral':
        arch_string = f"{config['hparams']['farch']}"  # default name
    else:
        ValueError(f"mode_hemis {mode_hemis} not recognized")

    exp_name = f"{config['exp_name']}-{config['hparams']['mode_hemis']}-{arch_string}-{config['hparams']['mode_heads']}"
    date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    version = f"{date_time}-seed{seed}"

    return save_dir, exp_name, version

def add_results_dual(seed, result, runs, accuracies):
    rdict = {
        'seed': seed,
        'result': result
    }
    runs.append(rdict)
    
    acc_fine = result[0]['test_acc_fine']
    acc_coarse = result[0]['test_acc_coarse']
    accuracies.append((acc_fine, acc_coarse))

def add_results_single(seed, result, runs, accuracies):
    rdict = {
        'seed': seed,
        'result': result
    }
    runs.append(rdict)
    
    acc = result[0]['test_acc']
    accuracies.append(acc)

def collect_results_dual(runs, accuracies):

    accuracies_fine = [acc[0] for acc in accuracies]
    accuracies_coarse = [acc[1] for acc in accuracies]

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
    return results

def collect_results_single(runs, accuracies):
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
    return results

def main(config_path) -> None:

    config = run_cli(config_path=config_path)
    seeds = config['seeds']
    mode_heads = config['hparams']['mode_heads']
    mode_out = config['hparams']['mode_out']
    mode_hemis = config['hparams']['mode_hemis']

    runs, accuracies, checkpoints = [], [], []               # one for each seed
    for seed in seeds:
        print(f"Running seed {seed}")

        if seed is not None:
            pl.seed_everything(seed)

        save_dir, exp_name, version = get_exp_names(config, seed)        
        logger = TensorBoardLogger(save_dir=save_dir, name=exp_name, version=version)

        monitor_var = config['ckpt_callback']['monitor']
        ckpt_callback = ModelCheckpoint(
            filename='{epoch}-{' + monitor_var + ':.3f}',
            **config['ckpt_callback'],
        )

        if 'callbacks' in config['trainer_params']:
            config['trainer_params']['callbacks'] = yaml_func(config['trainer_params']['callbacks'])
        if config['trainer_params']['default_root_dir'] == "None":
            config['trainer_params']['default_root_dir'] = osp.dirname(__file__)
   
        if mode_heads == 'both':
            model = SupervisedLightningModuleDualHead(config)
        else:
            model = SupervisedLightningModuleSingleHead(config)

        trainer = pl.Trainer(**config['trainer_params'],
                            callbacks=[ckpt_callback],
                            logger=logger)
                            # limit_train_batches=0.1)


        imdm = DataModule(mode_heads,
                          train_dir=config['dataset']['train_dir'],
                          test_dir=config['dataset']['test_dir'],
                          raw_data_dir=config['dataset'].get('raw_data_dir', None),
                          batch_size=config['hparams']['batch_size'],
                          num_workers=config['hparams']['num_workers'])

        if mode_hemis != 'ensemble':
            trainer.fit(model, datamodule=imdm)
            
            # get the path to the best checkpoint saved by the callback
            checkpoint_path = ckpt_callback.best_model_path
            checkpoints.append(checkpoint_path)

        # save the config file in the results folder
        dest_dir = os.path.join(save_dir, exp_name, version)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.copy(config_path, f'{dest_dir}/config.yaml')

        if config["evaluate"]:
            result = trainer.test(model, datamodule=imdm)

            if mode_out == 'both':
                add_results_dual(seed, result, runs, accuracies)
            else:
                add_results_single(seed, result, runs, accuracies)

    if config["evaluate"]:
        
        if mode_out == 'both':
            results = collect_results_dual(runs, accuracies)
        else:
            results = collect_results_single(runs, accuracies)

        # write the results list to a yaml file
        with open(f'{dest_dir}/results.yaml', 'w') as f:
            yaml.dump(results, f)

    return checkpoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual hemisphere training/testing')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                    help='Path to the base config file for training macro-arch with 2 heads. Relative to the folder where you ran this from.')
    args = parser.parse_args()
    main(args.config)
