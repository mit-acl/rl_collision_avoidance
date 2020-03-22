#!/bin/bash
set -e

function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Disable all tensorflow warnings/info (keep errors)
export TF_CPP_MIN_LOG_LEVEL=2

# Directory of this script
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR=${THIS_DIR}
source $BASE_DIR/venv/bin/activate
echo "Entered virtualenv."

export PYTHONPATH=${BASE_DIR}/venv/lib/python3.5/site-packages