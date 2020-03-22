# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

class Config:
    #########################################################################
    # GENERAL PARAMETERS
    game_grid, game_ale, game_collision_avoidance = range(3) # Initialize game types as enum
    GAME_CHOICE         = game_collision_avoidance # Game choice: Either "game_grid" or "game_ale" or "game_collision_avoidance"
    DISPLAY_SCREEN      = False # Enable screen display during playback
    USE_IMAGE           = False # Enable image input
    NET_ARCH            = 'NetworkVP_rnn' # Neural net architecture
    ALL_ARCHS           = ['NetworkVP_rnn'] # Can add more model types here
    TRAIN_MODELS        = True # Enable to train
    LOAD_CHECKPOINT     = True # Load old models. Throws if the model doesn't exist
    PLAY_MODE           = False # Enable to see the trained agent in action (for testing)
    EVALUATE_MODE       = False # Enable to see the trained agent in action (for testing)
    DEBUG               = False # Enable debug (prints more information for debugging purpose)
    NORMALIZE_INPUT     = True
    USE_DROPOUT         = False
    WEIGHT_SHARING      = False
    USE_REGULARIZATION  = True
    LOAD_FROM_BACKUP_DIR= False
    RECORD_DATA_MODE    = False
    LOAD_FROM_WANDB_RUN_ID = 'run-20190817_074248-4qw1fcd9'

    AGENT_SORTING_METHOD = "closest_last"
    # AGENT_SORTING_METHOD = "closest_first"
    # AGENT_SORTING_METHOD = "time_to_impact"

    # SOCIAL_NORMS = "right"
    # SOCIAL_NORMS = "left"
    SOCIAL_NORMS = "none"

    # LOAD_EPISODE        = 2360000 # this is one of the files on the cadrl_ros github
    # LOAD_EPISODE        = 1653000 # this is one of the files on the cadrl_ros github
    # LOAD_EPISODE        = 2830001
    # LOAD_EPISODE        = 1900000 # 2018-IROS GA3C-10
    # LOAD_EPISODE        = 1491000 # 2018-IROS GA3C-4 (only trained on 2-4 agents... does not exist?)
    # LOAD_EPISODE        = 1239000 # trained from regression w/ 2-4 agents (took 12hrs)
    # LOAD_EPISODE        = 1972000 # retrained on 6-28-19 w/ 10 agents (starting from 1239000, took 18hrs for last step)
    # LOAD_EPISODE        = 1490000
    LOAD_EPISODE        = 0

    SIGMA = 0.0

    #########################################################################
    # COLLISION AVOIDANCE PARAMETER
    if GAME_CHOICE == game_collision_avoidance:
        MAX_NUM_AGENTS_IN_ENVIRONMENT = 20
        MAX_NUM_AGENTS_TO_SIM = 4
        NUM_TEST_CASES = 500
        PLOT_EPISODES = False # with matplotlib, plot after each episode
        PLOT_EVERY_N_EPISODES = 100 # for tensorboard visualization
        DT             = 0.2 # seconds between simulation time steps
        REWARD_AT_GOAL = 1.0 # Number of agents trying to get from start -> goal positions
        REWARD_COLLISION = -0.25 # Number of agents trying to get from start -> goal positions
        REWARD_GETTING_CLOSE   = -0.1 # Number of agents trying to get from start -> goal positions
        REWARD_NORM_VIOLATION   = -0.05 # Number of agents trying to get from start -> goal positions
        NUM_AGENT_STATES = 4 # Number of states (pos_x,pos_y,...)
        OTHER_OBS_LENGTH = 7 # number of states about another agent in observation vector
        NUM_STEPS_IN_OBS_HISTORY = 1 # number of time steps to store in observation vector
        NUM_PAST_ACTIONS_IN_STATE = 0
        COLLISION_DIST = 0.0 # meters between agents' boundaries for collision
        GETTING_CLOSE_RANGE = 0.2 # meters between agents' boundaries for collision
        STACKED_FRAMES = 1 # Num of inputs to DQN
        REWARD_MIN     = -100 # Reward Clipping
        REWARD_MAX     = 100 # Reward Clipping
        MAX_ITER       = 40 # Max iteration (time limit)
        TIMER_DURATION = 0.01 # In second visualization time for each step
        TRAIN_WITH_REGRESSION = False # Start training with regression phase before RL
        LOAD_REGRESSION = True # Initialize training with regression network
        MULTI_AGENT_ARCHS = ['RNN','WEIGHT_SHARING','VANILLA']
        # MULTI_AGENT_ARCH = 'VANILLA'
        # MULTI_AGENT_ARCH = 'WEIGHT_SHARING'
        MULTI_AGENT_ARCH = 'RNN'

        SENSING_HORIZON  = np.inf
        # SENSING_HORIZON  = 0.1

    #########################################################################
    # NUMBER OF AGENTS, PREDICTORS, TRAINERS, AND OTHER SYSTEM SETTINGS
    # IF THE DYNAMIC CONFIG IS ON, THESE ARE THE INITIAL VALUES
    AGENTS                        = 32 # Number of Agents
    PREDICTORS                    = 2 # Number of Predictors
    TRAINERS                      = 2 # Number of Trainers
    DEVICE                        = '/cpu:0' # Device
    DYNAMIC_SETTINGS              = False # Enable the dynamic adjustment (+ waiting time to start it)
    DYNAMIC_SETTINGS_STEP_WAIT    = 20
    DYNAMIC_SETTINGS_INITIAL_WAIT = 10

    #########################################################################
    # ALGORITHM PARAMETER
    DISCOUNT                = 0.97 # Discount factor
    TIME_MAX                = int(4/DT) # Tmax
    MAX_QUEUE_SIZE          = 100 # Max size of the queue
    PREDICTION_BATCH_SIZE   = 128
    IMAGE_WIDTH             = 84 # Input of the DNN
    IMAGE_HEIGHT            = 84
    EPISODES                = 1500000 # Total number of episodes and annealing frequency
    ANNEALING_EPISODE_COUNT = 1500000

    # OPTIMIZER PARAMETERS
    OPT_RMSPROP, OPT_ADAM   = range(2) # Initialize optimizer types as enum
    OPTIMIZER               = OPT_ADAM # Game choice: Either "game_grid" or "game_ale"
    LEARNING_RATE_REGRESSION_START = 4e-5 # Learning rate
    LEARNING_RATE_REGRESSION_END = 4e-5 # Learning rate
    LEARNING_RATE_RL_START     = 2e-5 # Learning rate
    LEARNING_RATE_RL_END     = 2e-5 # Learning rate
    RMSPROP_DECAY           = 0.99
    RMSPROP_MOMENTUM        = 0.0
    RMSPROP_EPSILON         = 0.1
    BETA_START              = 1e-4 # Entropy regularization hyper-parameter
    BETA_END                = 1e-4
    USE_GRAD_CLIP           = False # Gradient clipping
    GRAD_CLIP_NORM          = 40.0
    LOG_EPSILON             = 1e-6 # Epsilon (regularize policy lag in GA3C)
    TRAINING_MIN_BATCH_SIZE = 100 # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower

    #########################################################################
    # LOG AND SAVE
    TENSORBOARD                  = True # Enable TensorBoard
    TENSORBOARD_UPDATE_FREQUENCY = 100 # Update TensorBoard every X training steps
    SAVE_MODELS                  = True # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY               = 10000 # Save every SAVE_FREQUENCY episodes
    PRINT_STATS_FREQUENCY        = 1 # Print stats every PRINT_STATS_FREQUENCY episodes
    STAT_ROLLING_MEAN_WINDOW     = 1000 # The window to average stats
    RESULTS_FILENAME             = 'results.txt'# Results filename
    NETWORK_NAME                 = 'network'# Network checkpoint name

    #########################################################################
    # MORE EXPERIMENTAL PARAMETERS 
    MIN_POLICY = 0.0 # Minimum policy

    WANDB_PROJECT_NAME = "ga3c_cadrl"


    HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1 # num other agents
    AGENT_ID_LENGTH = 1 # id
    IS_ON_LENGTH = 1 # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5]) # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0]) # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    # if MAX_NUM_AGENTS_IN_ENVIRONMENT == 2:
    #     # NN input:
    #     # [dist to goal, heading to goal, pref speed, radius, other px, other py, other vx, other vy, other radius, combined radius, distance between]
    #     MAX_NUM_OTHER_AGENTS_OBSERVED = 1
    #     OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
    #     HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
    #     FULL_STATE_LENGTH = HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
    #     FIRST_STATE_INDEX = 0
    #     MULTI_AGENT_ARCH = 'NONE'

    #     NN_INPUT_AVG_VECTOR = np.hstack([HOST_AGENT_AVG_VECTOR,OTHER_AGENT_AVG_VECTOR])
    #     NN_INPUT_STD_VECTOR = np.hstack([HOST_AGENT_STD_VECTOR,OTHER_AGENT_STD_VECTOR])


    if MAX_NUM_AGENTS_IN_ENVIRONMENT > 2:
        if MULTI_AGENT_ARCH == 'RNN':
            MAX_NUM_OTHER_AGENTS_OBSERVED = 19
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 1

            NN_INPUT_AVG_VECTOR = np.hstack([RNN_HELPER_AVG_VECTOR,HOST_AGENT_AVG_VECTOR,np.tile(OTHER_AGENT_AVG_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])
            NN_INPUT_STD_VECTOR = np.hstack([RNN_HELPER_STD_VECTOR,HOST_AGENT_STD_VECTOR,np.tile(OTHER_AGENT_STD_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])

        # elif MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
        #     MAX_NUM_OTHER_AGENTS_OBSERVED = 3
        #     OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH + IS_ON_LENGTH
        #     HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
        #     FULL_STATE_LENGTH = HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
        #     FIRST_STATE_INDEX = 0
            
        #     NN_INPUT_AVG_VECTOR = np.hstack([HOST_AGENT_AVG_VECTOR,np.tile(np.hstack([OTHER_AGENT_AVG_VECTOR,IS_ON_AVG_VECTOR]),MAX_NUM_OTHER_AGENTS_OBSERVED)])
        #     NN_INPUT_STD_VECTOR = np.hstack([HOST_AGENT_STD_VECTOR,np.tile(np.hstack([OTHER_AGENT_STD_VECTOR,IS_ON_STD_VECTOR]),MAX_NUM_OTHER_AGENTS_OBSERVED)])
            
    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH
    NN_INPUT_SIZE = FULL_STATE_LENGTH



    #     FULL_STATE_DIST_TO_GOAL_INDEX = 0
    #     FULL_STATE_HEADING_TO_GOAL_INDEX = 1
    #     FULL_STATE_PREF_SPEED_INDEX = 2
    #     FULL_STATE_RADIUS_INDEX = 3
    #     FULL_STATE_OTHER_PX_INDEX = 4
    #     FULL_STATE_OTHER_PY_INDEX = 5
    #     FULL_STATE_OTHER_VX_INDEX = 6
    #     FULL_STATE_OTHER_VY_INDEX = 7
    #     FULL_STATE_OTHER_RADIUS_INDEX = 8
    #     FULL_STATE_DIST_BETWEEN_INDEX = 9
    #     FULL_STATE_COMBINED_RADIUS_INDEX = 10
