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


import sys
import time
if sys.version_info >= (3,0): from queue import Queue
else: from Queue import Queue
import numpy as np
import scipy.misc as misc
from Config import Config
from Collision_Avoidance import CollisionAvoidance


class Environment:
    def __init__(self, id):
        self._set_env(id)
        
        self.nb_frames    = Config.STACKED_FRAMES
        self.frame_q      = Queue(maxsize=self.nb_frames)
        self.audio_q      = Queue(maxsize=self.nb_frames)
        self.total_reward = 0

        if Config.USE_AUDIO:
            self.previous_state = self.current_state = [None, None]
        else:
            self.previous_state = self.current_state = None

        # self.reset()

    def _set_env(self, id):
        if Config.GAME_CHOICE == Config.game_grid:
            self.game = Gridworld(id, Config.ENV_ROW, Config.ENV_COL, Config.PIXEL_SIZE, Config.MAX_ITER, Config.AGENT_COLOR, Config.TARGET_COLOR,
                                  Config.DISPLAY_SCREEN, Config.TIMER_DURATION, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, Config.STACKED_FRAMES, Config.DEBUG)
        elif Config.GAME_CHOICE == Config.game_collision_avoidance:
            self.game = CollisionAvoidance()
        else: 
            raise ValueError("[ ERROR ] Invalid choice of game. Check Config.py for choices")

    def _get_current_state(self):
        if Config.DEBUG: print '[ DEBUG ] Environment::_get_current_state()'

        if Config.GAME_CHOICE == Config.game_collision_avoidance:
            agent_states_ = np.array(self.frame_q.queue)
            return agent_states_

        else:
            if Config.USE_AUDIO:
                if not self.frame_q.full() or not self.audio_q.full():
                    return [None, None]

                audio_ = np.array(self.audio_q.queue)
                audio_ = np.transpose(audio_, [1, 2, 0])
                # TODO concatenate audio into 1 long image instead

            if not self.frame_q.full():
                return None

            image_ = np.array(self.frame_q.queue)
            if Config.USE_IMAGE == True:
                image_ = np.transpose(image_, [1, 2, 0]) # e.g., changes image from (1,84,84) to (84,84,1) 

            if Config.USE_AUDIO:
                return [image_, audio_]
            else:
                return image_

    def _update_frame_q(self, frame):
        if self.frame_q.full():
            self.frame_q.get()# Pop oldest frame
        self.frame_q.put(frame)
        if Config.DEBUG: print '[ DEBUG ] Environment::frame_q size is: {}'.format(self.frame_q.qsize())

    def _update_audio_q(self, audio):
        if self.audio_q.full():
            self.audio_q.get()# Pop oldest frame
        self.audio_q.put(audio)
        if Config.DEBUG: print '[ DEBUG ] Environment::audio_q size is: {}'.format(self.audio_q.qsize())

    def reset(self, test_case=None, alg='A3C'):
        if Config.DEBUG: print '[ DEBUG ] Environment::reset()'
        self.total_reward = 0
        self.frame_q.queue.clear()

        if Config.USE_AUDIO:
            self.audio_q.queue.clear() 
            
            image, audio = self.game.reset(test_case=test_case)
            self._update_frame_q(image)
            self._update_audio_q(audio)

            self.previous_state = self.current_state = [None, None]
        else:
            self._update_frame_q(self.game.reset(test_case=test_case, alg=alg))
            self.previous_state = self.current_state = None

    def step(self, action, pid, count):
        if Config.DEBUG: print '[ DEBUG ] Environment::step()'
        observations, rewards, which_agents_done, game_over = self.game.step(action, pid, count)
        self.total_reward += np.sum(rewards)

        if Config.GAME_CHOICE == Config.game_collision_avoidance:
            self.latest_observations = observations
            self._update_frame_q(observations[:,1:])
            
            # for agent_observation in observations:
            #     # only use host agent's observations for training
            #     if agent_observation[0] == 0:
            #         self._update_frame_q(agent_observation[1:])

        elif Config.USE_AUDIO:
            image = observation[0]
            audio = observation[1]

            self._update_frame_q(image)
            self._update_audio_q(audio)
        else:
            image = observation
            self._update_frame_q(image)

        self.previous_state = self.current_state
        self.current_state = self._get_current_state()

        return rewards, which_agents_done, game_over

    def print_frame_q(self):
        return self.frame_q.qsize()
