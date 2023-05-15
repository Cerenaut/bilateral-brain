import yaml
import sys
import os.path as osp
import argparse

import arch_single_head.trainer as tr_single
import arch_dual_head.trainer as tr_dual
from utils import run_cli, mod_filename


data = {'fine': {}, 'coarse': {}}
data['fine']['train'] = 'datasets/CIFAR100/train/fine'
data['fine']['test'] = 'datasets/CIFAR100/test/fine'
data['coarse']['train'] = 'datasets/CIFAR100/train/coarse'
data['coarse']['test'] = 'datasets/CIFAR100/test/coarse'
data['raw'] = '../datasets/cifar-100-python/'

# this will be populated in train_system.py
epochs_dict = {'fine': {}, 'coarse': {}, 'bilateral': {}}


def train_hemispheres(arch, base_config_path, num_seeds, epochs):
  print("-------- train Left and Right hemispheres ---------")
  print(f"-- arch = {arch} --")
  print("---------------------------------------------------")

  checkpoints_dict = {}
  for label_type in ['fine', 'coarse']:

    # open base config  
    abs_filpath = osp.abspath(base_config_path)
    doc = run_cli(config_path=abs_filpath)

    # customize params
    doc['seeds'] = list(range(num_seeds))
    doc['hparams']['mode_heads'] = label_type
    doc['hparams']['arch'] = arch
    doc['dataset']['train_dir'] = data[label_type]['train']
    doc['dataset']['val_dir'] = data[label_type]['test']
    doc['dataset']['test_dir'] = data[label_type]['test']
    doc['dataset']['raw_data_dir'] = data['raw']

    if epochs is not None:
      doc['trainer_params']['max_epochs'] = epochs_dict[label_type]

    # write the config
    new_config_path = mod_filename(base_config_path, f'config-{label_type}')
    with open(new_config_path, 'w') as out:
      yaml.safe_dump(doc, out)

    # run the experiment
    checkpoints = tr_single.main(new_config_path)
    checkpoints_dict[label_type] = checkpoints

  return checkpoints_dict['fine'], checkpoints_dict['coarse']


def train_bilateral(f_arch, f_checkpoints, c_arch, c_checkpoints, base_config_path, epochs):
  print("-------- train Bilateral hemispheres with saved checkpoints ---------")
  print(f"-- f_arch = {f_arch}, c_arch = {c_arch} --")
  print(f"-- f_out_folder = {f_checkpoints}, c_out_folder = {f_checkpoints} --")
  print("--------------------------------------------------------------------")

  abs_filpath = osp.abspath(base_config_path)
  doc = run_cli(config_path=abs_filpath)

  i = 0
  for f_checkpoint, c_checkpoint in zip(f_checkpoints, c_checkpoints):

    # customize params
    doc['seed'] = [i]                                     # new seed for each unique set of hemispheres
    doc['hparams']['farch'] = f_arch                      # architecture for fine
    doc['hparams']['carch'] = c_arch                      # architecture for coarse
    doc['hparams']['mode_output'] = 'both'                # output from both heads
    doc['hparams']['mode_hemis'] = 'bilateral'            # two hemispheres
    doc['hparams']['model_path_fine'] = f_checkpoint      # load fine hemisphere
    doc['hparams']['model_path_coarse'] = c_checkpoint    # load coarse hemisphere

    doc['dataset']['train_dir'] = data['fine']['train']
    doc['dataset']['val_dir'] = data['fine']['test']
    doc['dataset']['test_dir'] = data['fine']['test']
    doc['dataset']['raw_data_dir'] = data['raw']

    if epochs is not None:
      doc['trainer_params']['max_epochs'] = epochs_dict['bilateral']

    # write the config
    new_config_path = mod_filename(base_config_path, f'config-bilateral-{i}')
    with open(new_config_path, 'w') as out:
      yaml.safe_dump(doc, out)

    # run the experiment
    tr_dual.main(new_config_path)
    i += 1


def main(arch, single_head_base_config, dual_head_base_config, 
         no_bilateral=False, num_seeds=1, epochs=None,
         f_checkpoints='', c_checkpoints=''):
  
  # if we don't have saved checkpoints, train the hemispheres
  if f_checkpoints == '' or c_checkpoints == '':
    f_checkpoints, c_checkpoints = train_hemispheres(arch, single_head_base_config, num_seeds, epochs)

  # optionally train the whole bilateral architecture
  if not no_bilateral:
    train_bilateral(arch, f_checkpoints, arch, c_checkpoints, dual_head_base_config, epochs)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Optionally train/test hemispheres (if no checkpoints provided) \
                                    and then optionally train/test bilateral architecture with those hemispheres.')

  # add arguments
  parser.add_argument('--arch', type=str, default='resnet9', choices=['resnet9', 'vgg11'],
                      help='Architecture of the model')
  parser.add_argument('--sh_base_config', type=str, default='arch_single_head/configs/config.yaml',
                      help='Path to the base config file for training individual hemisphers (with single head). Relative to the current folder.')
  parser.add_argument('--dh_base_config', type=str, default='arch_dual_head/configs/config.yaml',
                      help='Path to the base config file for training the whole bilateral network (with dual heads).  Relative to the current folder.')
  parser.add_argument('--no_bilateral', dest='no_bilateral', action='store_true',
                      help='Train the hemispheres, but don\'t continue to the bilateral architecture.')
  parser.set_defaults(no_bilateral=False)
  parser.add_argument('--num_seeds', type=int, default='1',
                      help='The number of seeds to do for each hemisphere, and hence the bilateral architecture (there will be one for each trained pair of hemispheres). \
The seeds will be 0, 1, ..., num_seeds-1.')

  # add the argument for the list of integers
  parser.add_argument('--epochs', metavar='N', type=int, nargs='+',
                    help='List of integers, expecting 3 for `fine epochs`, `coarse epochs`, `bilateral epochs`')

  parser.add_argument('--f_chk', type=str, default='',
                      help='Path to folder of saved checkpoints for fine hemispheres. If fine and coarse checkpoints provided, then dont train hemispheres.')
  parser.add_argument('--c_chk', type=str, default='',
                      help='Path to folder of saved checkpoints for coarse hemispheres. If fine and coarse checkpoints provided, then dont train hemispheres.')
  
  # parse the command line arguments
  args = parser.parse_args()

  # access the arguments
  print(f"arch: {args.arch}")
  print(f"sh_base_config: {args.sh_base_config}")
  print(f"dh_base_config: {args.dh_base_config}")
  print(f"no_bilateral: {args.no_bilateral}")
  print(f"num_seeds: {args.num_seeds}")
  print(f"epochs: {args.epochs}")
  print(f"f_chk: {args.f_chk}")
  print(f"c_chk: {args.c_chk}")

  # TODO add code to go through checkpoint folders, and create an array of checkpoints for each hemisphere
  # in order to support passing f_chk and c_chk, rather than training the hemispheres here


  if args.epochs is not None:
    epochs_dict['fine'] = args.epochs[0]
    epochs_dict['coarse'] = args.epochs[1]
    epochs_dict['bilateral'] = args.epochs[2]

  main(args.arch, 
       args.sh_base_config, 
       args.dh_base_config, 
       args.no_bilateral, 
       args.num_seeds,
       args.epochs,
       args.f_chk, 
       args.c_chk)



# Train single hemispheres on fine and coarse, with 5 seeds each, with vgg11 backbone
# python train_system.py --no_bilateral --arch vgg11 --num_seeds 5 --epochs 200 200 100
