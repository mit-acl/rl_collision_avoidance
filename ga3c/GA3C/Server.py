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

import time
from multiprocessing import Queue
from GA3C import Config
from ProcessAgent import ProcessAgent
from NetworkVP_rnn import NetworkVP_rnn
from ProcessStats import ProcessStats
from NoThreadDynamicAdjustment import ThreadDynamicAdjustment
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer
from gym_collision_avoidance.envs.policies.GA3C_CADRL.network import VxVyDiscreteActions as CA_Actions
from Regression import Regression

class Server:
    def __init__(self):
        self.stats              = ProcessStats()
        self.training_q         = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q       = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        # TODO clean logic later
        if Config.GAME_CHOICE == Config.game_ale:
            self.num_actions = Config.NUM_ACTIONS # TODO Environment().get_num_actions() disables sounds for processors
        elif Config.GAME_CHOICE == Config.game_grid:
            gridworld_actions = Actions()
            self.num_actions = gridworld_actions.num_actions
        elif Config.GAME_CHOICE == Config.game_collision_avoidance:
            self.actions = CA_Actions()
            self.num_actions = self.actions.num_actions
        else:
            raise ValueError("[ ERROR ] Server game choice invalid!")

        print("[Server] Making model...")
        self.model = self.make_model()

        if Config.TRAIN_VERSION == Config.TRAIN_ONLY_REGRESSION:
            print("[Server] Training Regression.")
            Regression(self.model, Config.MAX_NUM_AGENTS_IN_ENVIRONMENT, self.actions).train_model()
            print("[Server] Finished training regression. Exiting.")
            assert(0)
        elif Config.TRAIN_VERSION == Config.LOAD_REGRESSION_THEN_TRAIN_RL:
            # if getattr(Config, "LOAD_FROM_WANDB_RUN_ID", False):
            #     print("[Server] Loading Regression Model then training RL.")
            self.model.load(learning_method='regression')
            # else:
            #     print("[Server] No LOAD_FROM_WANDB_RUN_ID, so just initializing a new NN instead.")
            self.stats.episode_count.value = 0
        elif Config.TRAIN_VERSION == Config.LOAD_RL_THEN_TRAIN_RL:
            print("[Server] Loading RL Checkpoint Model then training more RL.")
            self.stats.episode_count.value = self.model.load(learning_method='RL')

        self.training_step      = 0
        self.frame_counter      = 0
        self.agents             = []
        self.predictors         = []
        self.trainers           = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)# NOTE Server is passed in here.

    

    def make_model(self):
        if Config.NET_ARCH in Config.ALL_ARCHS:
            return globals()[Config.NET_ARCH](Config.DEVICE, Config.NETWORK_NAME, self.num_actions) # TODO can probably change Config.NETWORK_NAME to Config.NET_ARCH
        else:
            raise Exception('The model name %s does not exist' % Config.NET_ARCH)   
    
    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join()
        self.trainers.pop()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()

    def add_agent(self):
        self.agents.append(ProcessAgent(len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q, self.num_actions))
        self.agents[-1].start()

    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].join()
        self.agents.pop()

    def train_model(self, x_, r_, a_, trainer_id):
        # [ INFO ] x_.shape: (45, 84, 84, 4) <--> (batch_size, row, col, channel)
        self.model.train(x_, r_, a_, trainer_id)
        self.training_step += 1
        self.frame_counter += x_.shape[0]
        self.stats.training_count.value += 1

        # Tensorboard logging
        if Config.TENSORBOARD and self.stats.training_count.value % Config.TENSORBOARD_UPDATE_FREQUENCY == 0:
            reward, roll_reward = self.stats.return_reward_log()
            self.model.log(x_, r_, a_, reward, roll_reward, self.stats.episode_count.value)

    def save_model(self):
        self.model.save(self.stats.episode_count.value)

    def main(self):
        # Start Thread objects by calling start() methods
        self.stats.start()
        self.dynamic_adjustment.run()# NOTE self.dynamic_adjustment is NOT thread anymore

        # If Config.PLAY_MODE == True, disable trainers
        if Config.PLAY_MODE:
            for trainer in self.trainers:
                trainer.enabled = False

        # Algorithm parameters
        learning_rate_multiplier = (Config.LEARNING_RATE_RL_END - Config.LEARNING_RATE_RL_START) / Config.ANNEALING_EPISODE_COUNT
        beta_multiplier = (Config.BETA_END - Config.BETA_START) / Config.ANNEALING_EPISODE_COUNT

        while self.stats.episode_count.value < Config.EPISODES:
            # Linearly anneals the learning rate up to Config.ANNEALING_EPISODE_COUNT, after which it maintains at Config.LEARNING_RATE_END
            step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = Config.LEARNING_RATE_RL_START + learning_rate_multiplier * step
            self.model.beta = Config.BETA_START + beta_multiplier * step

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if Config.SAVE_MODELS and self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0
            time.sleep(0.01)

        # Terminate all with exit_flag == True
        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
            print("removed agent... {} left".format(len(self.agents)))
        # i = 0
        # while self.predictors:
        #     print("removing predictor... {}".format(i))
        #     i -= 1
        #     self.remove_predictor()
        # i = 0
        # while self.trainers:
        #     print("removing trainer... {}".format(i))
        #     i -= 1
        #     self.remove_trainer()
        print("all done.")
