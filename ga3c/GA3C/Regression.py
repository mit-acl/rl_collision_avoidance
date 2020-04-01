import numpy as np
import pickle
import time
from GA3C import Config
import os

class Regression():
    def __init__(self, model, num_agents, actions):
        self.model = model
        self.file_dir = os.path.dirname(os.path.realpath(__file__))+'/datasets'
        self.num_agents = num_agents
        self.actions = actions
        self.num_actions = actions.num_actions

        np.set_printoptions(suppress=True,precision=4)

    def train_model(self):
        self.train_model_with_regression()

    def pickle_load(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f,encoding='latin1')

    def load_ped_data(self):
        num_agents = 4

        if Config.MULTI_AGENT_ARCH == 'RNN':
            prepend = 'rnn_'
        else:
            prepend = ''

        file_name = self.file_dir+\
            "/2_3_4_agents_{prepend}cadrl_dataset_action_value_{mode}.p"

        dataset_ped_train = self.pickle_load(file_name.format(prepend=prepend, mode="train"))
        dataset_ped_test = self.pickle_load(file_name.format(prepend=prepend, mode="test"))
        num_train_pts = dataset_ped_train[0].shape[0]
        num_test_pts = dataset_ped_test[0].shape[0]
        print('dataset contains %d pts, training set has %d pts, test set has %d pts' % \
                (num_train_pts+num_test_pts, num_train_pts, num_test_pts))

        return dataset_ped_train, dataset_ped_test

    def train_model_with_regression(self):
        # Regression Phase (Supervised Learning)

        ##########################################
        # dataset_ped_train/test is some type of list of 3 elements:
        # [num training pts, nn state (e.g. 18)]
        # [num training pts, action (2)]
        # [num training pts, value (1)]
        ##########################################
        dataset_ped_train, dataset_ped_test = self.load_ped_data()

        # train neural network
        self.train(dataset_ped_train, dataset_ped_test)

    def one_hot(self, indices, max_range):
        num_inds = indices.shape[0]
        one_hot_vec = np.zeros((num_inds, max_range))
        one_hot_vec[np.arange(num_inds), indices] = 1
        return one_hot_vec

    def one_warm(self, indices, max_range):
        num_inds = indices.shape[0]
        amount_for_max = 0.8
        amount_for_others = 1.0 - amount_for_max
        one_hot_vec = (amount_for_others/(max_range-1))*np.ones((num_inds, max_range))
        one_hot_vec[np.arange(num_inds), indices] = amount_for_max
        return one_hot_vec

    def train(self, training_data, test_data):
        # training param
        batch_size = Config.REGRESSION_BATCH_SIZE
        num_training_steps = Config.REGRESSION_NUM_TRAINING_STEPS
        plot_step = Config.REGRESSION_PLOT_STEP

        # parse data
        # output action is [speed, how much to change current angle]
        input_x_cadrl = training_data[0]; output_action = training_data[1]; output_y = training_data[2]
        input_x_test_cadrl = test_data[0]; output_action_test = test_data[1]; output_y_test = test_data[2]
        nb_training_examples = input_x_cadrl.shape[0]
        nb_testing_examples = input_x_test_cadrl.shape[0]
        # convert to one hot representation
        
        # if np.shape(input_x_cadrl)[1] == 31:
        #     pref_speed = input_x_cadrl[:,1]
        # elif np.shape(input_x_cadrl)[1] == 18:
        #     pref_speed = input_x_cadrl[:,6]
        # else:
        #     print("number of states in regression dataset doesn't match 2 or 4 agent structure")

        # output_action[:,0] /= pref_speed_train # normalize commanded speed by pref_speed
        # output_action_test[:,0] /= pref_speed_test # normalize commanded speed by pref_speed
        output_action_indices = self.find_action_index(output_action, self.actions.actions)
        output_action_test_indices = self.find_action_index(output_action_test, self.actions.actions)
        output_action_one_hot = self.one_hot(output_action_indices, self.num_actions)
        output_action_test_one_hot = self.one_hot(output_action_test_indices, self.num_actions)
        # output_action_one_hot = self.one_warm(output_action_indices, self.num_actions)
        # output_action_test_one_hot = self.one_warm(output_action_test_indices, self.num_actions)

        # # convert cadrl state to new state
        # input_x = util.convert_cadrl_state_to_state(input_x_cadrl)
        # input_x_test = util.convert_cadrl_state_to_state(input_x_test_cadrl)
        
        input_x = input_x_cadrl
        input_x_test = input_x_test_cadrl
        # input_x = np.zeros((nb_training_examples,Config.NN_INPUT_SIZE))
        # input_x_test = np.zeros((nb_testing_examples,Config.NN_INPUT_SIZE))
        # input_x[:,:26] = input_x_cadrl
        # input_x_test[:,:26] = input_x_test_cadrl

        initial_step = 0
        total_steps = initial_step + num_training_steps
        for kk in range(initial_step, total_steps):
            if kk % plot_step == 0:
                print("Regression Training Step %d/%d" %(kk, total_steps))
                rand_ind = np.random.randint(0,nb_training_examples)
                rand_x = np.expand_dims(input_x[rand_ind,:], axis=0); rand_a = output_action_one_hot[rand_ind,:]; rand_y = output_y[rand_ind];
                print(rand_x)
                network_p, network_v = self.model.predict_p_and_v(rand_x)
                # util.plot_snapshot(rand_x, rand_a, rand_y, self.actions.actions, network_p, network_v, figure_name="regression_snapshot")
                
                minibatch_indices = np.random.choice(nb_testing_examples, min(nb_testing_examples, batch_size), replace=False)
                x = input_x_test[minibatch_indices]; a = output_action_test_one_hot[minibatch_indices]; y = np.squeeze(output_y_test[minibatch_indices])
                v_loss, p_loss, loss = self.model.get_regression_loss(x, y, a)
                print("[Regression] Loss on test set:", v_loss, p_loss, loss)

            minibatch_indices = np.random.choice(nb_training_examples, min(nb_training_examples, batch_size), replace=False)
            x = input_x[minibatch_indices]; a = output_action_one_hot[minibatch_indices]; y = np.squeeze(output_y[minibatch_indices])
            trainer_id = 0;
            self.model.train(x, y, a, trainer_id, learning_method='regression')

        # save checkpoint
        self.model.save(0, learning_method='regression')

    def find_action_index(self, actions, possible_actions):
        indices = np.zeros((actions.shape[0], ), dtype=np.int)

        # Complicated method
        actions_x = actions[:,0] * np.cos(actions[:,1])
        actions_y = actions[:,0] * np.sin(actions[:,1])
        possible_actions_x = possible_actions[:,0] * np.cos(possible_actions[:,1])
        possible_actions_y = possible_actions[:,0] * np.sin(possible_actions[:,1])
        diff_x = actions_x[:,np.newaxis] - possible_actions_x
        diff_y = actions_y[:,np.newaxis] - possible_actions_y
        dist_sq = diff_x ** 2 + diff_y ** 2
        indices = np.argmin(dist_sq, axis=1)


        # # stopping
        # zero_inds = np.where(actions[:,0] < 0.01)[0]
        # non_zero_inds = np.where(actions[:,0] > 0.01)[0]
        # indices[non_zero_inds] = np.argmin(dist_sq[non_zero_inds,:], axis=1)
        # angle_diff = abs(actions[:,1][:,np.newaxis] - possible_actions[:,1])
        # indices[zero_inds] = np.argmin(angle_diff[zero_inds,:], axis=1)

        # Simple method
        # angle_diff = abs(actions[:,1][:,np.newaxis] - possible_actions[:,1])
        # indices = np.argmin(angle_diff, axis=1)
        
        # for i in range(100):
        #     print(actions[i,:], possible_actions[indices[i],:])

        # print('actions', actions[:40,:])
        # print('indices (final)', indices[:40])
        # print('possible_actions[indices,:]', possible_actions[indices[:40],:])
        # assert(0)
        return indices

    def train_model_with_regression_old(self):
        # Regression Phase (Supervised Learning)

        ##########################################
        # dataset_ped_train/test is some type of list of 3 elements:
        # [num training pts, nn state (e.g. 18)]
        # [num training pts, action (2)]
        # [num training pts, value (1)]
        ##########################################
        dataset_ped_train, dataset_ped_test = self.load_ped_data()

        # print("There are %i training examples in total." %len(dataset_ped_train[0]))
        # small_angle_inds = np.where(abs(dataset_ped_train[0][:,5]) > np.pi/3)[0]
        # big_angle_inds = np.setdiff1d(np.arange(len(dataset_ped_train[0])),small_angle_inds)
        # print("There are %i training examples with heading error > pi/3, and %i other ones" %(len(small_angle_inds),len(big_angle_inds)))
        # small_angle_inds = np.where(abs(dataset_ped_train[0][:,5]) > 2.5*np.pi/3)[0]
        # print("There are %i training examples with heading error > 2pi/3" %len(small_angle_inds))

        # # For actions that involve big angle changes, set it to turn to the right. 
        # small_angle_inds = np.where(abs(dataset_ped_train[0][:,5]) < np.pi/3)[0]
        # big_angle_inds = np.setdiff1d(np.arange(len(dataset_ped_train[0])),small_angle_inds)
        # dataset_ped_train[1][big_angle_inds,1] = np.pi/3
        # dataset_ped_train[1][big_angle_inds,0] = 0.5
        # dataset_ped_train[1][small_angle_inds,1] = -dataset_ped_train[1][small_angle_inds,1]

        # # dataset_ped_train[0] = dataset_ped_train[0][big_angle_inds,:]
        # # dataset_ped_train[1] = dataset_ped_train[1][big_angle_inds,:]
        # # dataset_ped_train[2] = dataset_ped_train[2][big_angle_inds,:]

        # # remove actions where there's another agent nearby
        # no_neighbor_inds = np.where(np.linalg.norm(dataset_ped_train[0][:,9:11], axis=1) > 8.0)[0]
        # print("no_neighbor_inds:", no_neighbor_inds)
        # neighbor_inds = np.setdiff1d(np.arange(len(dataset_ped_train[0])),no_neighbor_inds)
        # dataset_ped_train[0] = dataset_ped_train[0][no_neighbor_inds,:]
        # dataset_ped_train[1] = dataset_ped_train[1][no_neighbor_inds,:]
        # dataset_ped_train[2] = dataset_ped_train[2][no_neighbor_inds,:]


        # print("heading:", dataset_ped_train[0][big_angle_inds,5])
        # print("action:", np.squeeze(dataset_ped_train[1][big_angle_inds,1]))
        # action_angle_change = util.find_angle_diff(dataset_ped_train[0][big_angle_inds,5], np.squeeze(dataset_ped_train[1][big_angle_inds,1]))
        # print(np.max(action_angle_change), np.min(action_angle_change))
        # TODO: Remove actions from regression dataset that involve non-max speed 



        ####################
        # CADRL dataset fixes
        #################

        # # For actions that are quite close to the goal, remove from dataset
        # near_goal_inds = np.where(abs(dataset_ped_train[0][:,Config.FIRST_STATE_INDEX]) < 0.5)[0]
        # far_from_goal_inds = np.setdiff1d(np.arange(len(dataset_ped_train[0])),near_goal_inds)

        # dataset_ped_train[0] = dataset_ped_train[0][far_from_goal_inds,:]
        # dataset_ped_train[1] = dataset_ped_train[1][far_from_goal_inds,:]
        # dataset_ped_train[2] = dataset_ped_train[2][far_from_goal_inds,:]

        # # For actions that are quite close to the goal, remove from dataset
        # near_goal_inds = np.where(abs(dataset_ped_test[0][:,Config.FIRST_STATE_INDEX]) < 0.5)[0]
        # far_from_goal_inds = np.setdiff1d(np.arange(len(dataset_ped_test[0])),near_goal_inds)

        # dataset_ped_test[0] = dataset_ped_test[0][far_from_goal_inds,:]
        # dataset_ped_test[1] = dataset_ped_test[1][far_from_goal_inds,:]
        # dataset_ped_test[2] = dataset_ped_test[2][far_from_goal_inds,:]

        # # For actions that have zero speed, remove from dataset
        # zero_speed_inds = np.where(abs(dataset_ped_train[1][:,:]) == np.array([0.0,0.0]))[0]
        # non_zero_speed_inds = np.setdiff1d(np.arange(len(dataset_ped_train[1])),zero_speed_inds)

        # dataset_ped_train[0] = dataset_ped_train[0][non_zero_speed_inds,:]
        # dataset_ped_train[1] = dataset_ped_train[1][non_zero_speed_inds,:]
        # dataset_ped_train[2] = dataset_ped_train[2][non_zero_speed_inds,:]

        # # For actions that have zero speed, remove from dataset
        # zero_speed_inds = np.where(abs(dataset_ped_test[1][:,:]) == np.array([0.0]))[0]
        # non_zero_speed_inds = np.setdiff1d(np.arange(len(dataset_ped_test[1])),zero_speed_inds)

        # dataset_ped_test[0] = dataset_ped_test[0][non_zero_speed_inds,:]
        # dataset_ped_test[1] = dataset_ped_test[1][non_zero_speed_inds,:]
        # dataset_ped_test[2] = dataset_ped_test[2][non_zero_speed_inds,:]

        

        # train neural network
        self.train(dataset_ped_train, dataset_ped_test)
