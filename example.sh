#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh

# Train tf 
print_header "Running example python script"

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
export GYM_CONFIG_PATH=$DIR/ga3c/GA3C/Config.py
export GYM_CONFIG_CLASS=Train

cd $DIR/ga3c/GA3C
# wandb off
python Run.py