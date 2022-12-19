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
from unsupervised import UnsupervisedLightningModule


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


def objective(trial: optuna.trial.Trial) -> float:
    config = run_cli()
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
    
    
    k = trial.suggest_float("k", 0.25, 0.75, step=.25)
    per_k = trial.suggest_categorical("per_k", [0.25, 0.5, 0.75])
    # lr = 1 / (10 ** trial.suggest_int("lr", 2, 4))
    lr = 1e-2
    
    config['hparams']['k'] = k
    config['hparams']['per_k'] = per_k
    config['hparams']['lr'] = lr
    
    config['logger']['version'] =f'res9|lr={lr}|bs=128|k={k}|%k={per_k}|linearsparse|'
        
    model = UnsupervisedLightningModule(config)
    logger = TestTubeLogger(
        config['logger']['save_dir'],
        name=config['logger']['name'],
        version=config['logger']['version'],)
    trainer = pl.Trainer(**config['trainer_params'],
                         callbacks=[ckpt_callback],
                         logger=logger,
                        #  resume_from_checkpoint='',
                         )
    imdm = DataModule(
        train_dir='/home/chandramouli/Documents/kaggle/CIFAR-100-Coarse/train',
        val_dir='/home/chandramouli/Documents/kaggle/CIFAR-100-Coarse/test',
        batch_size=config['hparams']['batch_size'])
    trainer.fit(model, datamodule=imdm)
    return trainer.callback_metrics["val_loss"].item()

def main() -> None:
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=10)

    string = "Number of finished trials: {}\n".format(str(len(study.trials)))

    string += "Best trial:\n"
    trial = study.best_trial

    string += "  Value: {}\n".format(str(trial.value))

    string += "  Params: \n"
    for key, value in trial.params.items():
        string += "    {}: {}".format(key, str(value))
    
    with open('./trial_results.txt', 'w') as f:
        f.write(string)
        f.close()


if __name__ == '__main__':
    cli_lightning_logo()
    main()
