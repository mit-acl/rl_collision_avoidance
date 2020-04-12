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

from gym_collision_avoidance.envs.config import Config as EnvConfig

class Train(EnvConfig):
    def __init__(self):
        ### PARAMETERS THAT OVERWRITE/IMPACT THE ENV'S PARAMETERS
        if not hasattr(self, "MAX_NUM_AGENTS_IN_ENVIRONMENT"):
            self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        if not hasattr(self, "MAX_NUM_AGENTS_TO_SIM"):
            self.MAX_NUM_AGENTS_TO_SIM = 4

        # self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'laserscan']
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agents_states']
        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        self.MULTI_AGENT_ARCH_RNN, self.MULTI_AGENT_ARCH_WEIGHT_SHARING, self.MULTI_AGENT_ARCH_LASERSCAN = range(3)
        self.MULTI_AGENT_ARCH = self.MULTI_AGENT_ARCH_WEIGHT_SHARING

        if self.MULTI_AGENT_ARCH == self.MULTI_AGENT_ARCH_WEIGHT_SHARING:
            self.MAX_NUM_OTHER_AGENTS_OBSERVED = 7
        elif self.MULTI_AGENT_ARCH in [self.MULTI_AGENT_ARCH_RNN, self.MULTI_AGENT_ARCH_LASERSCAN]:
            self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        ### INITIALIZE THE ENVIRONMENT'S PARAMETERS
        EnvConfig.__init__(self)

        ### GENERAL PARAMETERS
        self.game_grid, self.game_ale, self.game_collision_avoidance = range(3) # Initialize game types as enum
        self.GAME_CHOICE         = self.game_collision_avoidance # Game choice: Either "game_grid" or "game_ale" or "game_collision_avoidance"
        self.USE_WANDB = True
        self.WANDB_PROJECT_NAME = "ga3c_cadrl"
        self.DEBUG               = False # Enable debug (prints more information for debugging purpose)
        self.RANDOM_SEED_1000 = 0 # np.random.seed(this * 1000 + env_id)

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

        self.LOAD_RL_THEN_TRAIN_RL, self.TRAIN_ONLY_REGRESSION, self.LOAD_REGRESSION_THEN_TRAIN_RL = range(3)

        ### NETWORK
        self.NET_ARCH            = 'NetworkVP_rnn' # Neural net architecture
        self.ALL_ARCHS           = ['NetworkVP_rnn'] # Can add more model types here
        self.NORMALIZE_INPUT     = True
        self.USE_DROPOUT         = False
        self.USE_REGULARIZATION  = True

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
        self.MIN_POLICY = 0.0 # Minimum policy

        # OPTIMIZER PARAMETERS
        self.OPT_RMSPROP, self.OPT_ADAM   = range(2) # Initialize optimizer types as enum
        self.OPTIMIZER               = self.OPT_ADAM # Game choice: Either "game_grid" or "game_ale"
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
        self.SAVE_FREQUENCY               = 50000 # Save every SAVE_FREQUENCY episodes
        self.SPECIAL_EPISODES_TO_SAVE = [] # Save these episode numbers, in addition to ad SAVE_FREQUENCY 
        self.PRINT_STATS_FREQUENCY        = 1 # Print stats every PRINT_STATS_FREQUENCY episodes
        self.STAT_ROLLING_MEAN_WINDOW     = 1000 # The window to average stats
        self.RESULTS_FILENAME             = 'results.txt'# Results filename
        self.NETWORK_NAME                 = 'network'# Network checkpoint name

class TrainPhase1(Train):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        self.MAX_NUM_AGENTS_TO_SIM = 4
        Train.__init__(self)
        self.TRAIN_VERSION = self.LOAD_REGRESSION_THEN_TRAIN_RL
        if self.MULTI_AGENT_ARCH == self.MULTI_AGENT_ARCH_RNN:
            self.LOAD_FROM_WANDB_RUN_ID = 'run-rnn'
        elif self.MULTI_AGENT_ARCH == self.MULTI_AGENT_ARCH_WEIGHT_SHARING:
            self.LOAD_FROM_WANDB_RUN_ID = 'run-ws-'+str(self.MAX_NUM_OTHER_AGENTS_OBSERVED+1)
        self.EPISODE_NUMBER_TO_LOAD = 0

        self.EPISODES                = 1500000 # Total number of episodes and annealing frequency
        self.ANNEALING_EPISODE_COUNT = 1500000

        self.SPECIAL_EPISODES_TO_SAVE = [1490000, 1500000]

class TrainPhase2(Train):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 10
        self.MAX_NUM_AGENTS_TO_SIM = 10
        Train.__init__(self)
        self.EPISODES                = 2000000
        self.ANNEALING_EPISODE_COUNT = 2000000
        self.TRAIN_VERSION = self.LOAD_RL_THEN_TRAIN_RL
        # self.LOAD_FROM_WANDB_RUN_ID = 'run-20200401_205620-2dfp6yeg'
        # self.EPISODE_NUMBER_TO_LOAD        = 1450000
        self.LOAD_FROM_WANDB_RUN_ID = 'run-20200324_221727-2tz70xqi'
        self.EPISODE_NUMBER_TO_LOAD        = 1490000
        self.SPECIAL_EPISODES_TO_SAVE = [1990000, 2000000]

class TrainRegression(Train):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 2

        Train.__init__(self)
        self.TRAIN_VERSION = self.TRAIN_ONLY_REGRESSION
        self.REGRESSION_BATCH_SIZE = 2000
        self.REGRESSION_NUM_TRAINING_STEPS = 3000
        self.REGRESSION_PLOT_STEP = 100
        self.LEARNING_RATE_REGRESSION_START = 4e-5 # Learning rate
        self.LEARNING_RATE_REGRESSION_END = 4e-5 # Learning rate

        self.DATASET_NAME = "_"
        
        # Laserscan mode
        if self.MULTI_AGENT_ARCH == self.MULTI_AGENT_ARCH_LASERSCAN:
            self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'laserscan']
            self.DATASET_NAME = "laserscan_"