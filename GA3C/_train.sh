# ./_clean.sh
# ./_tensorboard.sh &

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # Get directory of this script

# Collision-Avoidance python module                                                                              
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Installing Collision-Avoidance python module"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
cd $DIR/../../environment/Collision-Avoidance
python -m pip install -I .

# Train tf 
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Training network"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
cd $DIR
mkdir checkpoints > /dev/null 2>&1
mkdir logs > /dev/null 2>&1
python GA3C.py "$@"
