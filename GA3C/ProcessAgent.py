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
import numpy as np
from datetime import datetime
from multiprocessing import Process, Queue, Value
from Config import Config
from Environment import Environment
from Experience import Experience
import pickle

np.set_printoptions(precision=4)


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q, num_actions):
        super(ProcessAgent, self).__init__()
        self.id              = id
        self.prediction_q    = prediction_q
        self.training_q      = training_q
        self.episode_log_q   = episode_log_q
        self.num_actions     = num_actions
        self.actions         = np.arange(self.num_actions)
        self.discount_factor = Config.DISCOUNT
        self.wait_q          = Queue(maxsize=1)
        self.exit_flag       = Value('i', 0)
        self.count           = 1

    # @staticmethod
    def _accumulate_rewards(self, experiences, discount_factor, terminal_reward, done):
        reward_sum = terminal_reward # terminal_reward is called R in a3c paper

        returned_exp = experiences[:-1] # Returns all but final experience in most cases. Final exp saved for next batch. 
        leftover_term_exp = None # For special case where game finishes but with 1 experience longer than TMAX
        n_exps = len(experiences)-1 # Does n_exps-step backward updates on all experiences

        # Exception case for experiences length of 0
        if len(experiences) == 1:
            experiences[0].reward = np.clip(experiences[0].reward, Config.REWARD_MIN, Config.REWARD_MAX) 
            return experiences, leftover_term_exp 
        else:
            if done and len(experiences) == Config.TIME_MAX+1:
                leftover_term_exp = [experiences[-1]]
            if done and len(experiences) != Config.TIME_MAX+1:
                n_exps = len(experiences)
                returned_exp = experiences

            for t in reversed(xrange(0, n_exps)):
                # experiences[t].reward is single-step reward here
                r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX) 
                reward_sum = discount_factor * reward_sum + r 
                #  experiences[t]. reward now becomes y_r (target reward, with discounting), and is used as y_r in training thereafter. I.e., variable name is overloaded. Totally OK but dirty.
                experiences[t].reward = reward_sum 

            # Final experience is removed 
            return returned_exp, leftover_term_exp


    def convert_to_nparray(self, experiences):
        x_ = np.array([exp.state_image for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences], dtype=np.int32)].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])

        if Config.GAME_CHOICE == Config.game_collision_avoidance:
            return x_, r_, a_
        else:
            return x_, None, r_, a_

    def predict(self, current_state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, current_state[1:]))

        # wait for the prediction to come back
        p, v = self.wait_q.get()

        return p, v

    def select_action(self, prediction):
        if Config.PLAY_MODE or Config.EVALUATE_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self):
        # Initialize
        self.env.reset()
        game_over   = False
        experiences = [[] for i in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)]
        updated_exps = [None for i in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)]
        updated_leftover_exps = [None for i in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)]
        time_counts  = np.zeros((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))
        reward_sum_logger  = np.zeros((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))
        which_agents_done_and_trained  = np.full((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT), False, dtype=bool)

        while not game_over:
            # Initial step
            if self.env.current_state is None:
                if Config.DEBUG: print('[ DEBUG ] ProcessAgent::Initial step')
                self.env.step(-1, self.pid, self.count)# Action 0 corresponds to null action
                # self.count += 1
                continue

            if Config.GAME_CHOICE == Config.game_collision_avoidance:
                actions = np.empty((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))
                predictions = np.empty((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT,self.env.game.actions.num_actions))
                values = np.empty((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))
                # print("self.env.latest_observation:", self.env.latest_observation)
                for i, agent_observation in enumerate(self.env.latest_observations):
                    if i not in self.env.game.which_agents_running_ga3c: continue
                    # Prediction
                    # print("[ProcessAgent]", "i:", i, "agent_observation:", agent_observation)
                    # p, v = self.predict(agent_observation)
                    prediction, value = self.predict(agent_observation)
                    # Select action
                    action = self.select_action(prediction)
                    
                    predictions[i] = prediction
                    values[i] = value
                    actions[i] = action

                    # print("action", actions[i])
                # print("actions:", actions)
                # Take action --> Receive reward, done (and also store self.env.previous_state for access below)
                rewards, which_agents_done, game_over = self.env.step(actions, self.pid, self.count)

                # Only use 1st agent's experience for learning # TODO: Use 2nd agent too


            # else:
            #     # Prediction
            #     prediction, value = self.predict(self.env.current_state)

            #     # Select action
            #     action = self.select_action(prediction)

            #     # Take action --> Receive reward, done (and also store self.env.previous_state for access below)
            #     reward, done = self.env.step(action, self.pid, self.count)

            for i in range(len(which_agents_done)): # Loop through all feedback from environment (which may not be equal to Config.MAX_NUM_AGENTS)
                if i not in self.env.game.which_agents_running_ga3c: continue
                # Reward
                reward_sum_logger[i] += rewards[i]

                prediction = predictions[i]
                value = values[i]
                action = actions[i]
                reward = rewards[i]
                done = which_agents_done[i]
                # Add to experience
                if Config.GAME_CHOICE == Config.game_collision_avoidance:
                    exp = Experience(self.env.previous_state[0,i,:], None,
                                     action, prediction, reward, done)
                else:
                    exp = Experience(self.env.previous_state, None,
                                     action, prediction, reward, done)

                if Config.DEBUG: print('[ DEBUG ] ProcessAgent::previous_state.shape: {}'.format(np.shape(self.env.previous_state)))

                experiences[i].append(exp)

                # If episode is done
                # Config.TIME_MAX controls how often data is yielded/sent back to the for loop in the run(). 
                # It is used to ensure, for games w long episodes, that data is sent back to the trainers sufficiently often
                # The shorter Config.TIME_MAX is, the more often the data queue is updated 
                if which_agents_done[i] or time_counts[i] == Config.TIME_MAX and which_agents_done_and_trained[i] == False:
                    if which_agents_done[i]:
                        terminal_reward = 0
                        which_agents_done_and_trained[i] = True
                    else:
                        terminal_reward = value
                    updated_exps[i], updated_leftover_exps[i] = self._accumulate_rewards(experiences[i], self.discount_factor, terminal_reward, which_agents_done[i])

                    if Config.GAME_CHOICE == Config.game_collision_avoidance:
                        x_, r_, a_ = self.convert_to_nparray(updated_exps[i])# NOTE if Config::USE_AUDIO == False, audio_ is None
                        yield x_, r_, a_, reward_sum_logger[i] / self.env.game.num_agents_running_ga3c # sends back data without quitting the current fcn
                    else:
                        x_, audio_, r_, a_ = self.convert_to_nparray(updated_exps) # NOTE if Config::USE_AUDIO == False, audio_ is None
                        yield x_, audio_, r_, a_, init_rnn_state, reward_sum_logger # Sends back data and starts here next time fcn is called

                    reward_sum_logger[i] = 0.0 # NOTE total_reward_logger in self.run() accumulates reward_sum_logger, so it is correct to reset it here 

                    if updated_leftover_exps[i] is not None:
                        #  terminal_reward = 0
                        if Config.GAME_CHOICE == Config.game_collision_avoidance:
                            x_, r_, a_ = self.convert_to_nparray(updated_leftover_exps[i]) # NOTE if Config::USE_AUDIO == False, audio_ is None
                            yield x_, r_, a_, reward_sum_logger[i] # TODO minor figure out what to send back in terms of rnn_state. Technically should be rnn_state[-1].
                        else:
                            x_, audio_, r_, a_ = self.convert_to_nparray(updated_leftover_exps) # NOTE if Config::USE_AUDIO == False, audio_ is None
                            yield x_, audio_, r_, a_, init_rnn_state, reward_sum_logger # TODO minor figure out what to send back in terms of rnn_state. Technically should be rnn_state[-1].

                    # Reset the tmax count
                    time_counts[i] = 0

                    # Keep the last experience for the next batch
                    experiences[i] = [experiences[i][-1]]

                time_counts[i] += 1
            self.count += 1


    def run_test_case(self, test_case=None, alg='A3C'):
        # Initialize
        self.env.reset(test_case=test_case, alg=alg)
        game_over   = False
        experiences = [[] for i in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)]
        updated_exps = [None for i in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)]
        updated_leftover_exps = [None for i in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)]
        time_counts  = np.zeros((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))
        reward_sum_logger  = np.zeros((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))
        which_agents_done_and_trained  = np.full((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT), False, dtype=bool)

        while not game_over:
            # Initial step
            if self.env.current_state is None:
                if Config.DEBUG: print('[ DEBUG ] ProcessAgent::Initial step')
                self.env.step(-1, self.pid, self.count)# Action 0 corresponds to null action
                self.count += 1
                continue

            if Config.GAME_CHOICE == Config.game_collision_avoidance:
                actions = np.empty((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))
                predictions = np.empty((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT,self.env.game.actions.num_actions))
                values = np.empty((Config.MAX_NUM_AGENTS_IN_ENVIRONMENT))
                # print("self.env.latest_observation:", self.env.latest_observation)
                for i, agent_observation in enumerate(self.env.latest_observations):
                    # Prediction
                    # print("[ProcessAgent]", "i:", i, "agent_observation:", agent_observation)
                    # p, v = self.predict(agent_observation)
                    prediction, value = self.predict(agent_observation)
                    # Select action
                    # print("prediction:", prediction)
                    action = self.select_action(prediction)
                    
                    predictions[i] = prediction
                    values[i] = value
                    actions[i] = action

                    # print("action", actions[i])
                # assert(0)
                # print("actions:", actions)
                # Take action --> Receive reward, done (and also store self.env.previous_state for access below)
                rewards, which_agents_done, game_over = self.env.step(actions, self.pid, self.count)
        time_to_goal = np.array([self.env.game.agents[i].t for i in range(len(self.env.game.agents))])
        extra_time_to_goal = np.array([self.env.game.agents[i].t - self.env.game.agents[i].straight_line_time_to_reach_goal for i in range(len(self.env.game.agents))])
        collision = np.array(np.any([self.env.game.agents[i].in_collision for i in range(len(self.env.game.agents))])).tolist()
        all_at_goal = np.array(np.all([self.env.game.agents[i].is_at_goal for i in range(len(self.env.game.agents))])).tolist()
        return time_to_goal, extra_time_to_goal, collision, all_at_goal

    def run(self):
        # Randomly sleep up to 1 second. Helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 5000 + self.id * 10))

        # NOTE env is created in here
        self.env = Environment(self.id)

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            count = 0

            if Config.GAME_CHOICE == Config.game_collision_avoidance:
                if Config.EVALUATE_MODE:

                    # algs = ['A3C','CADRL']
                    algs = ['A3C']
                    # algs = ['CADRL']
                    stats = {}
                    for alg in algs:
                        stats[alg] = {}
                        stats[alg]['non_collision_inds'] = []
                        stats[alg]['all_at_goal_inds'] = []
                        stats[alg]['stuck_inds'] = []
                    time_to_goal_sum = 0.0
                    num_test_cases_with_collision = 0
                    for test_case in range(Config.NUM_TEST_CASES):
                    # for test_case in range(5):
                        for alg in algs:
                            times_to_goal, extra_times_to_goal, collision, all_at_goal = self.run_test_case(test_case,alg=alg)
                            stats[alg][test_case] = {}
                            stats[alg][test_case]['times_to_goal'] = times_to_goal
                            stats[alg][test_case]['extra_times_to_goal'] = extra_times_to_goal
                            stats[alg][test_case]['mean_extra_time_to_goal'] = np.mean(extra_times_to_goal)
                            stats[alg][test_case]['total_time_to_goal'] = np.sum(times_to_goal)
                            stats[alg][test_case]['collision'] = collision
                            stats[alg][test_case]['all_at_goal'] = all_at_goal
                            if not collision: stats[alg]['non_collision_inds'].append(test_case)
                            if all_at_goal: stats[alg]['all_at_goal_inds'].append(test_case)
                            if not collision and not all_at_goal: stats[alg]['stuck_inds'].append(test_case)



                            print("Test Case:", test_case)
                            if collision:
                                print("*******Collision*********")
                                num_test_cases_with_collision += 1
                            if not collision and not all_at_goal:
                                print("*******Stuck*********")

                            print("Agents Time to goal:", times_to_goal)
                            print("Agents Extra Times to goal:", extra_times_to_goal)
                            print("Total time to goal (all agents):", np.sum(times_to_goal))
                            # time_to_goal_sum += np.sum(times_to_goal)
                            # print("Running total:", time_to_goal_sum)

                        # for x_, r_, a_, reward_sum in self.run_test_case(test_case):
                        #     if Config.DEBUG: print('[ DEBUG ] ProcessAgent::x_.shape is: {}'.format(x_.shape))
                        #     if len(x_.shape) > 1:
                        #         total_reward += reward_sum
                        #         total_length += len(r_) + 1  # +1 for last frame that we drop
                        #     else: 
                        #         print('[ DEBUG ] x_ has wrong shape of {}'.format(x_.shape))
                        #         import sys; sys.exit()
                        # self.episode_log_q.put((datetime.now(), total_reward, total_length))
                    print('-----------')
                    num = ''
                    if alg == 'A3C' and Config.LOAD_EPISODE == 1491000:
                        num = '1'
                    elif alg == 'A3C' and Config.LOAD_EPISODE == 1900000:
                        num = '2'
                    # pickle.dump(stats,open("logs/results/"+algs[0]+num+'.p','wb'))
                    # print('dumped')
                    
                    # # print(stats)
                    # non_collision_inds = np.intersect1d(stats['A3C']['non_collision_inds'], stats['CADRL']['non_collision_inds'])
                    # all_at_goal_inds = np.intersect1d(stats['A3C']['all_at_goal_inds'], stats['CADRL']['all_at_goal_inds'])
                    # no_funny_business_inds = np.intersect1d(non_collision_inds, all_at_goal_inds)
                    # for alg in algs:
                    #     print("Algorithm: %s" %alg)
                    #     num_collisions = test_case+1-len(stats[alg]['non_collision_inds'])
                    #     num_stuck = len(stats[alg]['stuck_inds'])
                    #     print("Total # test cases with collision: %i of %i (%.2f%%)" %(num_collisions,test_case+1,(100.0*num_collisions/(test_case+1))))
                    #     print("Total # test cases where agent got stuck: %i of %i (%.2f%%)" %(num_stuck,test_case+1,(100.0*num_stuck/(test_case+1))))
                    #     time_to_goal_sum = 0.0
                    #     extra_time_to_goal_sum = 0.0
                    #     extra_time_to_goal_vec = None
                    #     mean_extra_time_to_goal_list = []
                    #     for ind in no_funny_business_inds:
                    #         time_to_goal_sum += stats[alg][ind]['total_time_to_goal']
                    #         if extra_time_to_goal_vec is None:
                    #             extra_time_to_goal_vec = np.array(stats[alg][ind]['extra_times_to_goal'])
                    #         else:
                    #             extra_time_to_goal_vec = np.hstack([extra_time_to_goal_vec, np.array(stats[alg][ind]['extra_times_to_goal'])])
                    #         mean_extra_time_to_goal_list.append(stats[alg][ind]['mean_extra_time_to_goal'])
                    #     print("%s: total time to goal (non-collision/non-stuck cases): %.2f" %(alg, time_to_goal_sum))
                    #     print("%s: extra time to goal (non-collision/non-stuck cases):" %(alg))
                    #     print(np.percentile(extra_time_to_goal_vec,[50,75,90]))
                    #     print("%s: mean extra time to goal (non-collision/non-stuck cases):" %(alg))
                    #     print(np.percentile(np.array(mean_extra_time_to_goal_list),[50,75,90]))



                    break

                else:
                    for x_, r_, a_, reward_sum in self.run_episode():
                        if Config.DEBUG: print('[ DEBUG ] ProcessAgent::x_.shape is: {}'.format(x_.shape))
                        if len(x_.shape) > 1:
                            total_reward += reward_sum
                            total_length += len(r_) + 1  # +1 for last frame that we drop
                            self.training_q.put((x_, r_, a_))# NOTE if Config::USE_AUDIO == False, audio_ is None
                        else: 
                            print('[ DEBUG ] x_ has wrong shape of {}'.format(x_.shape))
                            import sys; sys.exit()


            else:
                for x_, audio_, r_, a_, reward_sum in self.run_episode():
                    if Config.DEBUG: print('[ DEBUG ] ProcessAgent::x_.shape is: {}'.format(x_.shape))
                    if len(x_.shape) > 1:
                        total_reward += reward_sum
                        total_length += len(r_) + 1  # +1 for last frame that we drop
                        self.training_q.put((x_, audio_, r_, a_))# NOTE if Config::USE_AUDIO == False, audio_ is None
                    else: 
                        print('[ DEBUG ] x_ has wrong shape of {}'.format(x_.shape))
                        import sys; sys.exit()

            self.episode_log_q.put((datetime.now(), total_reward, total_length))
