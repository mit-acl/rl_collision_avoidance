#!/bin/bash
set -e

MAKE_VENV=${1:-true}
SOURCE_VENV=${2:-true}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if $MAKE_VENV; then
    # Virtualenv w/ python3
    export PYTHONPATH=/usr/bin/python3 # point to your python3
    python3 -m pip install zipp==1.2.0 # virtualenv for python3.5
    python3 -m pip install virtualenv
    cd $DIR
    virtualenv -p python3 venv
fi

if $SOURCE_VENV; then
    echo "Sourcing venv"
    cd $DIR
    source venv/bin/activate
    export PYTHONPATH=${DIR}/venv/lib/python3.5/site-packages
fi

$DIR/gym-collision-avoidance/install.sh false false

# # Install this pkg and its requirements
python -m pip install -r requirements.txt
python -m pip install -e ga3c

# export PYTHONPATH=/home/mfe/code/carrl/venv/lib/python3.5/site-packages

echo "GA3C-CADRL was sucessfully installed."
