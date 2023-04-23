import yaml
import sys
import os.path as osp

import trainer as tr
from utils import run_cli

''' 
This is a script to train the Left and Right hemispheres.
'''

def main():
  print("-------- running train_hemispheres ---------")

  data = { 
    'fine': {},
    'coarse': {}
  }
  data['fine']['train'] = '../datasets/CIFAR100/train/fine'
  data['fine']['test'] = '../datasets/CIFAR100/test/fine'
  data['coarse']['train'] = '../datasets/CIFAR100/train/coarse'
  data['coarse']['test'] = '../datasets/CIFAR100/test/coarse'

  for label_type in ['fine', 'coarse']:

    print(f"------ label_type = {label_type}")

    # open base config  
    fullpath = osp.abspath('configs/config.yaml')
    doc = run_cli(config_path=fullpath)

    # customize params
    doc['logger']['name'] = label_type
    doc['hparams']['mode'] = 'narrow' if label_type == 'fine' else 'broad'
    doc['dataset']['train_dir'] = data[label_type]['train']
    doc['dataset']['val_dir'] = data[label_type]['test']
    doc['dataset']['test_dir'] = data[label_type]['test']

    # write the config
    new_config_path = osp.abspath(f'configs/config-{label_type}.yaml')
    with open(new_config_path, 'w') as out:
      yaml.dump(doc, out)

    # run the experiment
    tr.main(new_config_path)

  

if __name__ == '__main__':
  main()
