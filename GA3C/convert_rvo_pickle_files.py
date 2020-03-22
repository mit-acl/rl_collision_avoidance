import numpy as np
import pickle

test_cases = pickle.load(open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/test_cases/%s_agents_%i_cases.p" %(4, 500), "rb"))
print [a.tolist() for a in test_cases[102]]




# # value_train = pickle.load(open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/dataset_value_train.p", "rb"))
# # regr_train = pickle.load(open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/dataset_regr_train.p", "rb"))
# # value_test = pickle.load(open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/dataset_value_test.p", "rb"))
# # regr_test = pickle.load(open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/dataset_regr_test.p", "rb"))

# # regr_train.append(value_train[1])
# # regr_test.append(value_test[1])

# # pickle.dump(regr_train, open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/dataset_action_value_train.p", "wb"))
# # pickle.dump(regr_test, open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/dataset_action_value_test.p", "wb"))


# # convert from VANILLA/WEIGHT_SHARING state vector to RNN state vector (remove is_ons and add seq_len)
# x_train = pickle.load(open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/2_3_4_agents_cadrl_dataset_action_value_train.p", "rb"))
# x_test = pickle.load(open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/2_3_4_agents_cadrl_dataset_action_value_test.p", "rb"))
# states_train = x_train[0]
# states_test = x_test[0]
# fixed_states_train = np.zeros((np.shape(states_train)[0], 26))
# fixed_states_test = np.zeros((np.shape(states_test)[0], 26))

# other_agent_is_on_train_inds = [11,19,27]
# which_agents_on_train = states_train[:,other_agent_is_on_train_inds]
# num_agents_train = np.sum(which_agents_on_train, axis=1)

# mask_train = np.repeat(which_agents_on_train[:], 7, axis=1)
# other_agents_states_train = np.hstack([states_train[:,4:11],states_train[:,12:19],states_train[:,20:27]])
# masked_other_agents_states_train = other_agents_states_train * mask_train

# fixed_states_train[:,0] = num_agents_train
# fixed_states_train[:,1:5] = states_train[:,0:4] # agent 0
# fixed_states_train[:,5:26] = masked_other_agents_states_train

# other_agent_is_on_test_inds = [11,19,27]
# which_agents_on_test = states_test[:,other_agent_is_on_test_inds]
# num_agents_test = np.sum(which_agents_on_test, axis=1)

# mask_test = np.repeat(which_agents_on_test[:], 7, axis=1)
# other_agents_states_test = np.hstack([states_test[:,4:11],states_test[:,12:19],states_test[:,20:27]])
# masked_other_agents_states_test = other_agents_states_test * mask_test

# fixed_states_test[:,0] = num_agents_test
# fixed_states_test[:,1:5] = states_test[:,0:4] # agent 0
# fixed_states_test[:,5:26] = masked_other_agents_states_test




# # fixed_states_train[:,1:5] = states_train[:,0:4] # agent 0
# # fixed_states_train[:,5:12] = states_train[:,4:11] # agent 1
# # fixed_states_train[:,12:19] = states_train[:,12:19] # agent 2
# # fixed_states_train[:,19:26] = states_train[:,20:27] # agent 3

# # fixed_states_test[:,0] = num_agents_test
# # fixed_states_test[:,1:5] = states_test[:,0:4] # agent 0
# # fixed_states_test[:,5:12] = states_test[:,4:11] # agent 1
# # fixed_states_test[:,12:19] = states_test[:,12:19] # agent 2
# # fixed_states_test[:,19:26] = states_test[:,20:27] # agent 3

# fixed_x_train = x_train
# fixed_x_train[0] = fixed_states_train
# fixed_x_test = x_test
# fixed_x_test[0] = fixed_states_test

# pickle.dump(fixed_x_train, open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/2_3_4_agents_rnn_cadrl_dataset_action_value_train.p", "wb"))
# pickle.dump(fixed_x_test, open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/CADRL/pickle_files/multi/4_agents/2_3_4_agents_rnn_cadrl_dataset_action_value_test.p", "wb"))
