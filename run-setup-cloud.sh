#!/bin/bash

echo
echo "##############       run-cloud.sh       ###############"
echo "#                                                     #"
echo "# NOTE: Will run from the script's containing folder  #"
echo "#                                                     #"
echo "#######################################################"

if [ "$1" == "-h" -o "$1" == "--help" ]; then
  echo "Usage: `basename $0` HOST KEY_FILE USER PORT"
  exit 0
fi

host=${1:-localhost}
json=${2:-configs/config_unilateral_specialize.json}

echo "Using host = " $host
echo "Run script = " $json

# bash sync-cloud.sh $host

ssh -p 22 ubuntu@${host} 'bash --login -s' <<ENDSSH

  cd \$HOME
  
  git clone https://github.com/Cerenaut/bilateral-brain.git
  
  # setup data
  mkdir datasets
  cd datasets
  wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
  tar -xf cifar-100-python.tar.gz
  cd ../bilateral-brain
  python data_scripts/prepare_cifar.py

  # Install conda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  source ~/.bashrc
  conda create -n bilateral python=3.9

  # Install dependencies
  export RUN_DIR=\$HOME/bilateral-brain
  cd \$RUN_DIR
  
  conda activate bilateral
  sudo pip install -r requirements.txt

ENDSSH

status=$?

if [ $status -ne 0 ]
then
	echo "ERROR: Could not complete execute run-cloud.sh on remote machine through ssh." >&2
	echo "	Error status = $status" >&2
	echo "	Exiting now." >&2
	exit $status
fi
