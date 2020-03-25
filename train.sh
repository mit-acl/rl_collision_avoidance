#!/bin/bash
set -e


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh

# Train tf 
print_header "Running example python script"

export GYM_CONFIG_CLASS=${1:-TrainPhase2}
export GYM_CONFIG_PATH=${2:-$DIR/ga3c/GA3C/Config.py}

cd $DIR/ga3c/GA3C
# wandb off
python Run.py