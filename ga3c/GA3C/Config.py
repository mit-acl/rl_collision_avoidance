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


class TrainPhase1(EnvConfig):
    def __init__(self):
        ### PARAMETERS THAT OVERWRITE/IMPACT THE ENV'S PARAMETERS
        if not hasattr(self, "MAX_NUM_AGENTS_IN_ENVIRONMENT"):
            self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        if not hasattr(self, "MAX_NUM_AGENTS_TO_SIM"):
            self.MAX_NUM_AGENTS_TO_SIM = 4

        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agents_states']
        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        ### INITIALIZE THE ENVIRONMENT'S PARAMETERS
        EnvConfig.__init__(self)

        ### GENERAL PARAMETERS
        self.game_grid, self.game_ale, self.game_collision_avoidance = range(3) # Initialize game types as enum
        self.GAME_CHOICE         = self.game_collision_avoidance # Game choice: Either "game_grid" or "game_ale" or "game_collision_avoidance"
        self.USE_WANDB = False
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

        ### CHECKPOINT LOADING
        self.LOAD_FROM_WANDB_RUN_ID = 'run-dummy'
        # EPISODE_NUMBER_TO_LOAD        = 2360000 # this is one of the files on the cadrl_ros github
        # EPISODE_NUMBER_TO_LOAD        = 1653000 # this is one of the files on the cadrl_ros github
        # EPISODE_NUMBER_TO_LOAD        = 2830001
        # EPISODE_NUMBER_TO_LOAD        = 1900000 # 2018-IROS GA3C-10
        # EPISODE_NUMBER_TO_LOAD        = 1491000 # 2018-IROS GA3C-4 (only trained on 2-4 agents... does not exist?)
        # EPISODE_NUMBER_TO_LOAD        = 1239000 # trained from regression w/ 2-4 agents (took 12hrs)
        # EPISODE_NUMBER_TO_LOAD        = 1972000 # retrained on 6-28-19 w/ 10 agents (starting from 1239000, took 18hrs for last step)
        # EPISODE_NUMBER_TO_LOAD        = 1490000
        self.EPISODE_NUMBER_TO_LOAD        = 0


        self.LOAD_RL_THEN_TRAIN_RL, self.TRAIN_ONLY_REGRESSION, self.LOAD_REGRESSION_THEN_TRAIN_RL = range(3)
        self.TRAIN_VERSION = self.TRAIN_ONLY_REGRESSION
        # self.LOAD_RL_THEN_TRAIN_RL     = False
        # self.LOAD_NOTHING_THEN_TRAIN_REGRESSION_THEN_RL = False # Start from scratch, then train regression phase before RL
        # self.LOAD_REGRESSION_THEN_TRAIN_RL = False # Initialize training with regression network
        # self.TRAIN_ONLY_REGRESSION = True # Initialize training with regression network

        ### NETWORK
        self.NET_ARCH            = 'NetworkVP_rnn' # Neural net architecture
        self.ALL_ARCHS           = ['NetworkVP_rnn'] # Can add more model types here
        self.NORMALIZE_INPUT     = True
        self.USE_DROPOUT         = False
        self.USE_REGULARIZATION  = True

        # self.MULTI_AGENT_ARCH = 'RNN'
        self.MULTI_AGENT_ARCH = 'WEIGHT_SHARING'
        # self.MULTI_AGENT_ARCH = 'FC'

        #########################################################################
        # NUMBER OF AGENTS, PREDICTORS, TRAINERS, AND OTHER SYSTEM SETTINGS
        # IF THE DYNAMIC CONFIG IS ON, THESE ARE THE INITIAL VALUES
        self.AGENTS                        = 1 # Number of Agents
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
        self.SAVE_FREQUENCY               = 10000 # Save every SAVE_FREQUENCY episodes
        self.PRINT_STATS_FREQUENCY        = 1 # Print stats every PRINT_STATS_FREQUENCY episodes
        self.STAT_ROLLING_MEAN_WINDOW     = 1000 # The window to average stats
        self.RESULTS_FILENAME             = 'results.txt'# Results filename
        self.NETWORK_NAME                 = 'network'# Network checkpoint name

        self.EPISODES                = 1500000 # Total number of episodes and annealing frequency
        self.ANNEALING_EPISODE_COUNT = 1500000

class TrainPhase2(TrainPhase1):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 10
        self.MAX_NUM_AGENTS_TO_SIM = 10
        TrainPhase1.__init__(self)
        self.EPISODES                = 2000000
        self.ANNEALING_EPISODE_COUNT = 2000000
        self.LOAD_FROM_WANDB_RUN_ID = 'run-20200324_221727-2tz70xqi'
        self.LOAD_RL_THEN_TRAIN_RL     = True
        self.LOAD_REGRESSION_THEN_TRAIN_RL = False
        self.EPISODE_NUMBER_TO_LOAD        = 1490000

class TrainRegression(TrainPhase1):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        TrainPhase1.__init__(self)
        self.REGRESSION_BATCH_SIZE = 2000
        self.REGRESSION_NUM_TRAINING_STEPS = 3000
        self.REGRESSION_PLOT_STEP = 100

        self.LEARNING_RATE_REGRESSION_START = 4e-5 # Learning rate
        self.LEARNING_RATE_REGRESSION_END = 4e-5 # Learning rate


