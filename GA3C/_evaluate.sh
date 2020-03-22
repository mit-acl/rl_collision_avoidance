DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # Get directory of this script

# Collision-Avoidance python module                                                                              
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo "Installing Collision-Avoidance python module"
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
cd $DIR/../../environment/Collision-Avoidance
python -m pip install -I .

cd $DIR
python GA3C.py EVALUATE_MODE=True
