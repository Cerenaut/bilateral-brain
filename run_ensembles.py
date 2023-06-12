
from utils import run_cli, mod_filename
import os.path as osp
import trainer
import yaml


def main(base_config_path, e_models, ensemble_size, seeds):
    print("-------- Test ensemble ---------")
    print("---------------------------------")

    abs_filpath = osp.abspath(base_config_path)
    doc = run_cli(config_path=abs_filpath)

    for i in range(seeds):
        doc['seeds'] = [i]
        doc['hparams']['model_path_fine'] = e_models[i:i+ensemble_size] 

        print('-----------')
        print(f"seeds = {doc['seeds']}")
        print(f"model_path_fine = {doc['hparams']['model_path_fine']}")

        # write the config
        new_config_path = mod_filename(base_config_path, f'config-ensemble_{i}')
        with open(new_config_path, 'w') as out:
            yaml.safe_dump(doc, out)

        # run the experiment
        trainer.main(new_config_path)

if __name__ == '__main__':

    ensemble_models = './configs/ensemble_models.yaml'
    base_config_path = './configs/config_ensemble.yaml'
    ensemble_size = 2
    seeds = 5

    e_models = run_cli(config_path=ensemble_models)
    checkpoints = e_models['checkpoints']

    main(base_config_path, checkpoints, ensemble_size, seeds)
