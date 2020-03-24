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

from gym_collision_avoidance.envs.config import Config

class GA3CConfig(Config):
    def __init__(self):
        ### GA3C-SPECIFIC SETTINGS THAT INFLUENCE ENVIRONMENT CONFIGS
        # TODO: Find a way to force env config.py to inherit from a parent config.py

        Config.__init__(self)

        ### GENERAL PARAMETERS
        self.game_grid, self.game_ale, self.game_collision_avoidance = range(3) # Initialize game types as enum
        self.GAME_CHOICE         = self.game_collision_avoidance # Game choice: Either "game_grid" or "game_ale" or "game_collision_avoidance"
        self.USE_WANDB = False
        self.WANDB_PROJECT_NAME = "ga3c_cadrl"
        self.DEBUG               = False # Enable debug (prints more information for debugging purpose)

        ### OBSERVATIONS
        self.USE_IMAGE           = False # Enable image input
        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack([self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack([self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])
        self.FIRST_STATE_INDEX = 1
        self.HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH

        ### ACTIONS
        self.NUM_ACTIONS = 11

        ### CHECKPOINT LOADING
        self.LOAD_CHECKPOINT     = True # Load old models. Throws if the model doesn't exist
        self.LOAD_FROM_BACKUP_DIR= False
        self.LOAD_FROM_WANDB_RUN_ID = 'run-20190817_074248-4qw1fcd9'
        # LOAD_EPISODE        = 2360000 # this is one of the files on the cadrl_ros github
        # LOAD_EPISODE        = 1653000 # this is one of the files on the cadrl_ros github
        # LOAD_EPISODE        = 2830001
        # LOAD_EPISODE        = 1900000 # 2018-IROS GA3C-10
        # LOAD_EPISODE        = 1491000 # 2018-IROS GA3C-4 (only trained on 2-4 agents... does not exist?)
        # LOAD_EPISODE        = 1239000 # trained from regression w/ 2-4 agents (took 12hrs)
        # LOAD_EPISODE        = 1972000 # retrained on 6-28-19 w/ 10 agents (starting from 1239000, took 18hrs for last step)
        # LOAD_EPISODE        = 1490000
        self.LOAD_EPISODE        = 0
        self.TRAIN_WITH_REGRESSION = False # Start training with regression phase before RL
        self.LOAD_REGRESSION = True # Initialize training with regression network

        ### NETWORK
        self.NET_ARCH            = 'NetworkVP_rnn' # Neural net architecture
        self.ALL_ARCHS           = ['NetworkVP_rnn'] # Can add more model types here
        self.NORMALIZE_INPUT     = True
        self.USE_DROPOUT         = False
        self.WEIGHT_SHARING      = False
        self.USE_REGULARIZATION  = True
        self.MULTI_AGENT_ARCH = 'RNN'

        #########################################################################
        # NUMBER OF AGENTS, PREDICTORS, TRAINERS, AND OTHER SYSTEM SETTINGS
        # IF THE DYNAMIC CONFIG IS ON, THESE ARE THE INITIAL VALUES
        self.AGENTS                        = 32 # Number of Agents
        self.PREDICTORS                    = 2 # Number of Predictors
        self.TRAINERS                      = 2 # Number of Trainers
        self.DEVICE                        = '/cpu:0' # Device
        self.DYNAMIC_SETTINGS              = False # Enable the dynamic adjustment (+ waiting time to start it)
        self.DYNAMIC_SETTINGS_STEP_WAIT    = 20
        self.DYNAMIC_SETTINGS_INITIAL_WAIT = 10

        #########################################################################
        # ALGORITHM PARAMETER
        self.DISCOUNT                = 0.97 # Discount factor
        self.TIME_MAX                = int(4/self.DT) # Tmax
        self.MAX_QUEUE_SIZE          = 100 # Max size of the queue
        self.PREDICTION_BATCH_SIZE   = 128
        self.EPISODES                = 1500000 # Total number of episodes and annealing frequency
        self.ANNEALING_EPISODE_COUNT = 1500000
        self.MIN_POLICY = 0.0 # Minimum policy

        # OPTIMIZER PARAMETERS
        self.OPT_RMSPROP, self.OPT_ADAM   = range(2) # Initialize optimizer types as enum
        self.OPTIMIZER               = self.OPT_ADAM # Game choice: Either "game_grid" or "game_ale"
        self.LEARNING_RATE_REGRESSION_START = 4e-5 # Learning rate
        self.LEARNING_RATE_REGRESSION_END = 4e-5 # Learning rate
        self.LEARNING_RATE_RL_START     = 2e-5 # Learning rate
        self.LEARNING_RATE_RL_END     = 2e-5 # Learning rate
        self.RMSPROP_DECAY           = 0.99
        self.RMSPROP_MOMENTUM        = 0.0
        self.RMSPROP_EPSILON         = 0.1
        self.BETA_START              = 1e-4 # Entropy regularization hyper-parameter
        self.BETA_END                = 1e-4
        self.USE_GRAD_CLIP           = False # Gradient clipping
        self.GRAD_CLIP_NORM          = 40.0
        self.LOG_EPSILON             = 1e-6 # Epsilon (regularize policy lag in GA3C)
        self.TRAINING_MIN_BATCH_SIZE = 100 # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower

        #########################################################################
        # LOG AND SAVE
        self.TENSORBOARD                  = True # Enable TensorBoard
        self.TENSORBOARD_UPDATE_FREQUENCY = 100 # Update TensorBoard every X training steps
        self.SAVE_MODELS                  = True # Enable to save models every SAVE_FREQUENCY episodes
        self.SAVE_FREQUENCY               = 10000 # Save every SAVE_FREQUENCY episodes
        self.PRINT_STATS_FREQUENCY        = 1 # Print stats every PRINT_STATS_FREQUENCY episodes
        self.STAT_ROLLING_MEAN_WINDOW     = 1000 # The window to average stats
        self.RESULTS_FILENAME             = 'results.txt'# Results filename
        self.NETWORK_NAME                 = 'network'# Network checkpoint name


