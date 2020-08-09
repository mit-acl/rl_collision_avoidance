#!/bin/bash
set -e

MAKE_VENV=${1:-true}
SOURCE_VENV=${2:-true}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if $MAKE_VENV; then
    # Virtualenv w/ python3
    export PYTHONPATH=/usr/bin/python3 # point to your python3

    # Check that it's actually python 3.4-7
    PYTHONVERSION=`python --version`
    if [[ ${PYTHONVERSION:7:1} == "3" && (${PYTHONVERSION:9:1} == "4" || ${PYTHONVERSION:9:1} == "5" || ${PYTHONVERSION:9:1} == "6" || ${PYTHONVERSION:9:1} == "7") ]]; then
        echo "You have Python 3.[4-7] installed (${PYTHONVERSION}). Cool."
    else
        echo "Your python version is: ${PYTHONVERSION}. Please replace it with Python 3.4-3.7 so we can use tensorflow 1."
        exit 1
    fi

    python -m pip install zipp==1.2.0 # virtualenv for python3.5
    python -m pip install virtualenv
    cd $DIR
    python -m virtualenv venv
fi

if $SOURCE_VENV; then
    echo "Sourcing venv"
    cd $DIR
    source venv/bin/activate
    export PYTHONPATH=${DIR}/venv/lib/python3.5/site-packages
fi

# Install Git LFS (if not already)
if git lfs install | grep -q 'initialized'; then
    echo "Git LFS already installed"
else
    echo "Installing Git LFS and pulling"
    if [ "$(uname)" == "Darwin" ]; then
        # Do something under Mac OS X platform
        brew install git-lfs
        git lfs install
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Do something under GNU/Linux platform
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        sudo apt-get install git-lfs
        git lfs install
    fi
    git lfs pull
fi

$DIR/gym-collision-avoidance/install.sh false false


# # Install this pkg and its requirements
python -m pip install -r requirements.txt
python -m pip install -e ga3c

# export PYTHONPATH=/home/mfe/code/rl_collision_avoidance/venv/lib/python3.5/site-packages

echo "GA3C-CADRL was sucessfully installed."
