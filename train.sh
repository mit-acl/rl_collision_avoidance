#!/bin/bash
set -e


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh

export GYM_CONFIG_CLASS=${1:-TrainPhase2}
export GYM_CONFIG_PATH=${2:-$DIR/ga3c/GA3C/Config.py}

# Train tf 
print_header "Running GA3C-CADRL gym-collision-avoidance training script (${GYM_CONFIG_CLASS})"

cd $DIR/ga3c/GA3C
# wandb off
python Run.py