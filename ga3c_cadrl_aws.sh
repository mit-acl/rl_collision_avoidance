#!/bin/bash

# Be sure to grab mfe's ~/.tmux.conf before using this script,
# from: http://stahlke.org/dan/tmux-nested/

# To navigate:
# Shift == S , Control == C
# S-left/right, moves btwn local windows (bottom green bar)
# S-up/down, moves btwn local/remote tmux session within one local window

# If remote tmux session selected (S-up): 
# C-a <num>, will switch to remote window <num>
# C-a " , will create new horizontal split pane
# C-a c , make new remote window
# C-a q , show pane numbers
# C-a space, toggle btwn layouts

# How to use:
# 1) Replace IP addresses of your AWS instances
# 2) Start up all the aws windows/panes:
#		./ga3c_cadrl_aws.sh panes
# 3) C-a :setw synchronize-panes
# 4) Detach from local tmux session: C-a d


# to run:
# ./ga3c_cadrl_aws.sh panes


declare -a IPS=("ec2-107-22-23-253.compute-1.amazonaws.com" "ec2-3-90-86-95.compute-1.amazonaws.com")
NUM_IPS=${#IPS[@]}
MODE=$1

if [ -z "$1" ]
	then
		echo "No mode selected"
		echo "Example use: ./ga3c_cadrl_aws.sh start"
		exit
	else
		echo "Starting GA3C-CADRL: $MODE"
fi

AWS_KEY="$HOME/Downloads/mfe.pem" # Path of aws key
SESSION="ga3c_cadrl_aws"

if [ "$MODE" == "panes" ]; then
	# start local tmux session that everything will be nested inside of
	tmux new-session -s $SESSION -d

	for ((i=0; i<$NUM_IPS; i++)); do
		AWS_SSH_ADDRESS=${IPS[$i]}
		if [ "$i" -gt "0" ]; then
			tmux split-window -h;
		fi
		PANE_NUM=$i
		REMOTE_SESSION=${SESSION}_${PANE_NUM}

		# share mfe's tmux conf with remote server, then ssh in
		tmux send-keys -t $PANE_NUM "scp -i $AWS_KEY -o StrictHostKeyChecking=no ~/.tmux.conf ubuntu@$AWS_SSH_ADDRESS:" C-m
		tmux send-keys -t $PANE_NUM "ssh -L localhost:6022:localhost:6006 -L localhost:8822:localhost:8888 -i ${AWS_KEY} ubuntu@${AWS_SSH_ADDRESS}" C-m
		
		# start remote tmux session (nested within local tmux session)
		tmux send-keys -t $PANE_NUM "tmux new-session -s ${REMOTE_SESSION} -d" C-m
		
		# # create window layout you like
		# # tmux send-keys -t $WINDOW_NUM "tmux split-window -v" C-m
		# # tmux send-keys -t $WINDOW_NUM "tmux new-window" C-m
		# tmux send-keys -t $WINDOW_NUM "tmux select-window -t ${REMOTE_SESSION}:0" C-m
		
		# attach to remote tmux session
		tmux send-keys -t $PANE_NUM "tmux -2 attach-session -t ${REMOTE_SESSION} -d" C-m
	done

	tmux select-layout even-horizontal
	tmux -2 attach-session -t $SESSION
# elif [ "$MODE" == "panes_kill" ]; then
# 	EXTRA_SESSION=${SESSION}_extra
# 	tmux new-session -s $EXTRA_SESSION -d
# 	for ((i=0; i<$NUM_IPS; i++)); do
# 		AWS_SSH_ADDRESS=${IPS[$i]}
# 		if [ "$i" -gt "0" ]; then
# 			tmux split-window -h -t $EXTRA_SESSION;
# 		fi
# 		PANE_NUM=$i
# 		REMOTE_SESSION=${SESSION}_${PANE_NUM}

# 		# share mfe's tmux conf with remote server, then ssh in
# 		tmux send-keys -t $PANE_NUM "scp -i $AWS_KEY -o StrictHostKeyChecking=no ~/.tmux.conf ubuntu@$AWS_SSH_ADDRESS:" C-m
# 		tmux send-keys -t $PANE_NUM "ssh -L localhost:6022:localhost:6006 -L localhost:8822:localhost:8888 -i ${AWS_KEY} ubuntu@${AWS_SSH_ADDRESS}" C-m
		
# 		# start remote tmux session (nested within local tmux session)
# 		tmux send-keys -t $PANE_NUM "tmux kill-session -t ${REMOTE_SESSION}" C-m
		
# 	done
# 	tmux kill-session -t $EXTRA_SESSION
# elif [ "$MODE" == "panes_attach" ]; then
# 	# start local tmux session that everything will be nested inside of
# 	EXTRA_SESSION=${SESSION}_extra
# 	tmux new-session -s $EXTRA_SESSION -d

# 	for ((i=0; i<$NUM_IPS; i++)); do
# 		AWS_SSH_ADDRESS=${IPS[$i]}
# 		if [ "$i" -gt "0" ]; then
# 			tmux -t $EXTRA_SESSION split-window -h;
# 		fi
# 		PANE_NUM=$i
# 		REMOTE_SESSION=${SESSION}_${PANE_NUM}

# 		# share mfe's tmux conf with remote server, then ssh in
# 		tmux send-keys -t $PANE_NUM "scp -i $AWS_KEY -o StrictHostKeyChecking=no ~/.tmux.conf ubuntu@$AWS_SSH_ADDRESS:" C-m
# 		tmux send-keys -t $PANE_NUM "ssh -L localhost:6022:localhost:6006 -L localhost:8822:localhost:8888 -i ${AWS_KEY} ubuntu@${AWS_SSH_ADDRESS}" C-m
		
# 		# attach to remote tmux session
# 		tmux send-keys -t $PANE_NUM "tmux -2 attach-session -t ${REMOTE_SESSION}" C-m
# 	done

# 	tmux select-layout even-horizontal
# 	tmux -2 attach-session -t $EXTRA_SESSION

elif [ "$MODE" == "copy" ]; then
    for ((i=0; i<$NUM_IPS; i++)); do
        AWS_SSH_ADDRESS=${IPS[$i]}
        SAVE_DIR=~/ijrr_cadrl_results/multiple_seeds/ttc_order_${AWS_SSH_ADDRESS}
        mkdir ${SAVE_DIR}
        scp -r -i $AWS_KEY ubuntu@$AWS_SSH_ADDRESS:~/code/2017-avrl/src/tensorflow/GA3C/checkpoints/RL ${SAVE_DIR}/wandb
    done
fi
