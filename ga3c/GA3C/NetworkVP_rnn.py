# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
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

import os
import re
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from NetworkVPCore import NetworkVPCore
from GA3C import Config

class NetworkVP_rnn(NetworkVPCore):
    def __init__(self, device, model_name, num_actions):
        super(self.__class__, self).__init__(device, model_name, num_actions)

    def _create_graph(self):
        # Use shared parent class to construct graph inputs
        self._create_graph_inputs()

        # Put custom architecture here

        if Config.USE_REGULARIZATION:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
        else:
            regularizer = None

        if Config.NORMALIZE_INPUT:
            self.avg_vec = tf.constant(Config.NN_INPUT_AVG_VECTOR, dtype = tf.float32)
            self.std_vec = tf.constant(Config.NN_INPUT_STD_VECTOR, dtype = tf.float32)
            self.x_normalized = (self.x - self.avg_vec) / self.std_vec
        else:
            self.x_normalized = self.x


        self.num_other_agents = self.x[:,0]
        self.host_agent_vec = self.x_normalized[:,Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
        self.other_agent_vec = self.x_normalized[:,Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
        self.other_agent_seq = tf.reshape(self.other_agent_vec, [-1, Config.MAX_NUM_OTHER_AGENTS_OBSERVED, Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])

        if Config.MULTI_AGENT_ARCH == Config.MULTI_AGENT_ARCH_RNN:
            num_hidden = 64
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(num_hidden), self.other_agent_seq, dtype=tf.float32, sequence_length=self.num_other_agents)
            self.rnn_output = self.rnn_state.h
            self.layer1_input = tf.concat([self.host_agent_vec, self.rnn_output],1, name='layer1_input')

        elif Config.MULTI_AGENT_ARCH == Config.MULTI_AGENT_ARCH_WEIGHT_SHARING:
            ##############################################
            # Layer 1   - host agent gets its own set of weights
            #           - other agents share weights (since they are symmetric)
            ##############################################
            num_hidden_units_per_agent = 64
            self.other_agent_input = {}
            self.other_agent_is_on = {}
            for i in range(Config.MAX_NUM_OTHER_AGENTS_OBSERVED):
                self.other_agent_is_on[i] = tf.expand_dims(tf.cast(self.num_other_agents >= (i+1), dtype=tf.float32), axis=-1)
                self.other_agent_input[i] = tf.concat([self.other_agent_seq[:,i,:], self.other_agent_is_on[i]], axis=1)
                
                # Create dense filter for observation about 1 other agent
                if i == 0:
                    reuse = None
                else:
                    reuse = True
                other_agent_filter = tf.layers.dense(inputs=self.other_agent_input[i], units=num_hidden_units_per_agent, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'other_agent_layer1', reuse=reuse)

                # Concat the obs about that agent with the rest of them so far (if any)
                if i == 0:
                    self.other_agents_summary = other_agent_filter
                else:
                    self.other_agents_summary = tf.concat([self.other_agents_summary, other_agent_filter], axis=1)

            # Just like in the RNN, just stick the raw host_agent_vec on the front of the 
            self.layer1_input = tf.concat([self.host_agent_vec, self.other_agents_summary],1, name='layer1_input')

        # elif Config.MULTI_AGENT_ARCH == 'FC':
        #     self.layer1_input = tf.layers.dense(inputs=self.x_normalized[:,Config.FIRST_STATE_INDEX:], units=256, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'layer0')
        else:
            print("[NetworkVP_rnn.py] Config.MULTI_AGENT_ARCH is not a valid option.")
            assert(0)

        self.layer1 = tf.layers.dense(inputs=self.layer1_input, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'layer1')
        self.layer2 = tf.layers.dense(inputs=self.layer1, units=256, activation=tf.nn.relu, name = 'layer2')
        self.final_flat = tf.contrib.layers.flatten(self.layer2)
        
        # Use shared parent class to construct graph outputs/objectives
        self._create_graph_outputs()
