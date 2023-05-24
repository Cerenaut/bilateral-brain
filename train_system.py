import yaml
import sys
import os.path as osp
import argparse

from utils import run_cli, mod_filename
import trainer

data = {'fine': {}, 'coarse': {}}
data['train'] = 'datasets/CIFAR100/train'
data['test'] = 'datasets/CIFAR100/test'
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
    doc['dataset']['train_dir'] = data['train']
    doc['dataset']['val_dir'] = data['test']
    doc['dataset']['test_dir'] = data['test']
    doc['dataset']['raw_data_dir'] = data['raw']

    if epochs is not None:
      doc['trainer_params']['max_epochs'] = epochs_dict[label_type]

    # write the config
    new_config_path = mod_filename(base_config_path, f'config-{label_type}')
    with open(new_config_path, 'w') as out:
      yaml.safe_dump(doc, out)

    # run the experiment
    checkpoints = trainer.main(new_config_path)
    checkpoints_dict[label_type] = checkpoints

  return checkpoints_dict['fine'], checkpoints_dict['coarse']


def train_bilateral(f_arch, f_checkpoints, c_arch, c_checkpoints, base_config_path, epochs):
  print("-------- train Bilateral hemispheres with saved checkpoints ---------")
  print(f"-- f_arch = {f_arch}, c_arch = {c_arch} --")
  print(f"-- f_out_folder = {f_checkpoints}, c_out_folder = {f_checkpoints} --")
  print("--------------------------------------------------------------------")

  abs_filpath = osp.abspath(base_config_path)
  doc = run_cli(config_path=abs_filpath)

  for i, (f_checkpoint, c_checkpoint) in enumerate(zip(f_checkpoints, c_checkpoints)):

    # customize params
    doc['seeds'] = [i]                                     # new seed for each unique set of hemispheres
    doc['hparams']['farch'] = f_arch                      # architecture for fine
    doc['hparams']['carch'] = c_arch                      # architecture for coarse
    doc['hparams']['mode_output'] = 'both'                # output from both heads
    doc['hparams']['mode_hemis'] = 'bilateral'            # two hemispheres
    doc['hparams']['model_path_fine'] = f_checkpoint      # load fine hemisphere
    doc['hparams']['model_path_coarse'] = c_checkpoint    # load coarse hemisphere

    doc['dataset']['train_dir'] = data['train']
    doc['dataset']['val_dir'] = data['test']
    doc['dataset']['test_dir'] = data['test']
    doc['dataset']['raw_data_dir'] = data['raw']

    if epochs is not None:
      doc['trainer_params']['max_epochs'] = epochs_dict['bilateral']

    # write the config
    new_config_path = mod_filename(base_config_path, f'config-bilateral-{i}')
    with open(new_config_path, 'w') as out:
      yaml.safe_dump(doc, out)

    # run the experiment
    trainer.main(new_config_path)


def main(arch, single_head_base_config, dual_head_base_config, 
         no_bilateral=False, num_seeds=1, epochs=None,
         f_checkpoints='', c_checkpoints=''):
  
  # if we don't have saved checkpoints, train the hemispheres
  if f_checkpoints == None or c_checkpoints == None:
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
  parser.add_argument('--uni_base_config', type=str, default='arch_single_head/configs/config.yaml',
                      help='Path to the base config file for training individual hemisphers (with single head). Relative to the current folder.')
  parser.add_argument('--bi_base_config', type=str, default='arch_dual_head/configs/config.yaml',
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

  parser.add_argument('--trained_models', type=str, default='',
                      help='Path to yaml file of saved checkpoints (fine_checkpoints, coarse_checkpoints). If fine and coarse checkpoints provided, then dont train hemispheres.')
  

  # parse the command line arguments
  args = parser.parse_args()

  # access the arguments
  print(f"arch: {args.arch}")
  print(f"uni_base_config: {args.uni_base_config}")
  print(f"bi_base_config: {args.bi_base_config}")
  print(f"no_bilateral: {args.no_bilateral}")
  print(f"num_seeds: {args.num_seeds}")
  print(f"epochs: {args.epochs}")
  print(f"trained_models: {args.trained_models}")

  fine_checkpoints, coarse_checkpoints = None, None
  if args.trained_models != '':
    t_models = run_cli(config_path=args.trained_models)
    fine_checkpoints = t_models['fine_checkpoints']
    coarse_checkpoints = t_models['coarse_checkpoints']

  if args.epochs is not None:
    epochs_dict['fine'] = args.epochs[0]
    epochs_dict['coarse'] = args.epochs[1]
    epochs_dict['bilateral'] = args.epochs[2]

  main(args.arch, 
       args.uni_base_config, 
       args.bi_base_config, 
       args.no_bilateral, 
       args.num_seeds,
       args.epochs,
       fine_checkpoints,
       coarse_checkpoints)

