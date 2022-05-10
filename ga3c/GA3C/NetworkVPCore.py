import os
import re
import numpy as np
import tensorflow as tf
from GA3C import Config


class NetworkVPCore(object):
    def __init__(self, device, model_name, num_actions):

        # Set up wandb path (in case loading checkpt from certain runid)
        if Config.TRAIN_VERSION in [Config.LOAD_RL_THEN_TRAIN_RL, Config.LOAD_REGRESSION_THEN_TRAIN_RL]:
            learning_method = 'RL'
        elif Config.TRAIN_VERSION in [Config.TRAIN_ONLY_REGRESSION]:
            learning_method = 'regression'
        self.wandb_dir = os.path.dirname(os.path.realpath(__file__)) + '/checkpoints/' + learning_method

        # if training, add run to GA3C-CADRL project, add hyperparams and auto-upload checkpts
        if not Config.PLAY_MODE and not Config.EVALUATE_MODE and Config.USE_WANDB:
            import wandb
            from wandb.tensorflow import WandbHook
            wandb.init(project=Config.WANDB_PROJECT_NAME, dir=self.wandb_dir)
            for attr, value in Config.__dict__.items():
                # wandb can't handle np objs, but otherwise send it all
                if type(value) in [bool, int, float, list, str, tuple]:
                    wandb.config.update({attr: value})
            self.wandb_log = wandb.log
            self.checkpoints_save_dir = os.path.join(wandb.run.dir,"checkpoints")
            wandb.save(os.path.join(self.checkpoints_save_dir,"network*"), base_path=wandb.run.dir)
        else:
            self.checkpoints_save_dir = os.path.dirname(os.path.realpath(__file__)) + '/checkpoints/RL_tmp'

        # Initialize DNN TF computation graph
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions
        self.learning_rate_rl = Config.LEARNING_RATE_RL_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.compat.v1.Session(
                    graph=self.graph,
                    config=tf.compat.v1.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
                self.sess.run(tf.compat.v1.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_RL_THEN_TRAIN_RL or Config.LOAD_REGRESSION_THEN_TRAIN_RL or Config.SAVE_MODELS:
                    vars = tf.compat.v1.global_variables()
                    self.saver = tf.compat.v1.train.Saver({var.name: var for var in vars}, max_to_keep=0)

    
    def _create_graph_inputs(self):
            self.x = tf.compat.v1.placeholder(
                tf.float32, [None, Config.NN_INPUT_SIZE], name='X')
 
    def _create_graph_outputs(self):
        # FCN
        self.fc1 = tf.layers.dense(inputs=self.final_flat, units = 256, use_bias = True, activation=tf.nn.relu, name = 'fullyconnected1')

        # Cost: v 
        self.logits_v = tf.squeeze(tf.layers.dense(inputs=self.fc1, units = 1, use_bias = True, activation=None, name = 'logits_v'), axis=[1])
        self.y_r = tf.compat.v1.placeholder(tf.float32, [None], name='Yr')
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        # Cost: p
        self.logits_p = tf.layers.dense(inputs = self.fc1, units = self.num_actions, name = 'logits_p', activation = None)
        self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
        self.action_index = tf.compat.v1.placeholder(tf.float32, [None, self.num_actions])
        self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1, name='selection_action_prob')

        self.cost_p_advant= tf.compat.v1.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                    * (self.y_r - tf.stop_gradient(self.logits_v))  # Stop_gradient ensures the value gradient feedback doesn't contribute to policy learning
        self.var_beta = tf.compat.v1.placeholder(tf.float32, name='beta', shape=[])
        self.cost_p_entrop = -1. * self.var_beta * \
                    tf.reduce_sum(tf.compat.v1.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                  self.softmax_p, axis=1)

        self.cost_p_advant_agg = tf.reduce_sum(self.cost_p_advant, axis=0, name='cost_p_advant_agg')
        self.cost_p_entrop_agg = tf.reduce_sum(self.cost_p_entrop, axis=0, name='cost_p_entrop_agg')
        self.cost_p = -(self.cost_p_advant_agg + self.cost_p_entrop_agg)

        # Regression Cost Terms (policy)
        self.cost_p_sq_err = tf.nn.softmax_cross_entropy_with_logits(labels = self.action_index, logits = self.logits_p)
        self.cost_p_sq_err_agg = tf.reduce_sum(self.cost_p_sq_err, axis=0, name='cost_p_sq_err_agg')
        # self.cost_p_regression = self.cost_p_sq_err_agg - self.cost_p_entrop_agg
        self.cost_p_regression = self.cost_p_sq_err_agg
        self.cost_p_regression_sum = tf.reduce_sum(self.cost_p_regression)
        
        # Cost: total
        self.cost_all = self.cost_p + self.cost_v
        self.cost_regression = self.cost_p_regression + self.cost_v
        self.cost_regression_sum = tf.reduce_sum(self.cost_regression)

        # Optimizer
        self.var_learning_rate = tf.compat.v1.placeholder(tf.float32, name='lr', shape=[])
        if Config.OPTIMIZER == Config.OPT_RMSPROP:
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.var_learning_rate,
                                                    decay=Config.RMSPROP_DECAY,
                                                    momentum=Config.RMSPROP_MOMENTUM,
                                                    epsilon=Config.RMSPROP_EPSILON)
        elif Config.OPTIMIZER == Config.OPT_ADAM:
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        else:
            raise ValueError('Invalid optimizer chosen! Check Config.py!')

        # Grad clipping
        self.global_step = tf.Variable(0, trainable=False, name='step')
        if Config.USE_GRAD_CLIP:
            self.opt_grad = self.opt.compute_gradients(self.cost_all)
            self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
            self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
            
        else:
            self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)
            self.train_regression_op = self.opt.minimize(self.cost_regression, global_step=self.global_step)


    def _create_tensor_board(self):
        summaries = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)
        summaries.append(tf.compat.v1.summary.scalar("Pcost_advantage", self.cost_p_advant_agg))
        summaries.append(tf.compat.v1.summary.scalar("Pcost_entropy", self.cost_p_entrop_agg))
        summaries.append(tf.compat.v1.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.compat.v1.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.compat.v1.summary.scalar("cost_all", self.cost_all))
        summaries.append(tf.compat.v1.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.compat.v1.summary.scalar("Beta", self.var_beta))
        for var in tf.compat.v1.trainable_variables():
            summaries.append(tf.compat.v1.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.compat.v1.summary.histogram("activation_d2", self.fc1))
        summaries.append(tf.compat.v1.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.compat.v1.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.compat.v1.summary.merge(summaries)
        self.log_writer = tf.compat.v1.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate_rl}

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        return self.sess.run(self.logits_v, feed_dict={self.x: x})

    def predict_p(self, x):
        return self.sess.run(self.softmax_p, feed_dict={self.x: x})

    def predict_p_and_v(self, x):
        # feed_dict={self.x: x}
        # x, x_normalized, avg_vec, std_vec, num_other_agents, host_agent_vec, other_agent_vec, other_agent_seq, rnn_outputs, rnn_output, layer1_input = \
        #     self.sess.run([self.x, self.x_normalized, self.avg_vec, self.std_vec, self.num_other_agents, self.host_agent_vec, self.other_agent_vec, self.other_agent_seq, self.rnn_outputs, self.rnn_output, self.layer1_input], feed_dict=feed_dict)
        # print("x:", x[0:3,:])
        # print("self.avg_vec:", avg_vec)
        # print("self.std_vec:", std_vec)
        # print("x_normalized:", x_normalized[0:3,:])
        # print("num_other_agents:", num_other_agents)
        # print("host_agent_vec:", host_agent_vec[0:3,:])
        # print("other_agent_vec:", other_agent_vec[0:3,:])
        # print("other_agent_seq:", other_agent_seq[0:3,:,:])
        # print("rnn_outputs:", rnn_outputs[0:3,:,:])
        # print("rnn_output after:", rnn_output[0:3,:])
        # print("layer1_input:", layer1_input[0:3,:])
        # assert(0)
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})

    def train(self, x, y_r, a, trainer_id, learning_method='RL'): # TODO: Is trainer_id needed?
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        
        if learning_method == 'RL':
            feed_dict.update({self.var_learning_rate: self.learning_rate_rl})
            self.sess.run([self.train_op], feed_dict=feed_dict)

            # x, num_other_agents, host_agent_vec, other_agent_vec, other_agent_seq, rnn_outputs, rnn_output, layer1_input = \
                # self.sess.run([self.x, self.num_other_agents, self.host_agent_vec, self.other_agent_vec, self.other_agent_seq, self.rnn_outputs, self.rnn_output, self.layer1_input], feed_dict=feed_dict)
            # print("x:", x[0:3,:])
            # print("num_other_agents:", num_other_agents)
            # print("host_agent_vec:", host_agent_vec[0:3,:])
            # print("other_agent_vec:", other_agent_vec[0:3,:])
            # print("other_agent_seq:", other_agent_seq[0:3,:,:])
            # print("rnn_outputs:", rnn_outputs[0:3,:,:])
            # print("rnn_output after:", rnn_output[0:3,:])
            # print("layer1_input:", layer1_input[0:3,:])
            # assert(0)

        elif learning_method == 'regression':
            feed_dict.update({self.var_learning_rate: Config.LEARNING_RATE_REGRESSION_START})
            self.sess.run(self.train_regression_op, feed_dict=feed_dict)

    def log(self, x, y_r, a, reward, roll_reward, episode):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

        # Additionally log reward and rolling reward
        # Ref: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Reward", simple_value=reward)])
        self.log_writer.add_summary(summary, step)

        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="Roll_Reward", simple_value=roll_reward)])
        self.log_writer.add_summary(summary, step)

        if not Config.PLAY_MODE and not Config.EVALUATE_MODE and Config.USE_WANDB:
            self.wandb_log({'reward': reward, 'roll_reward': roll_reward, 'step': step, 'episode': episode})

    def _checkpoint_filename(self, episode, mode='save', learning_method='RL', wandb_runid_for_loading=None):
        if mode == 'save':
            d = self.checkpoints_save_dir
        elif mode == 'load':
            d = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints", learning_method, 'wandb', wandb_runid_for_loading, 'checkpoints')
        else:
            raise NotImplementedError

        path = os.path.join(d, '%s_%08d' % (self.model_name, episode))
        
        return path

    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[-1])

    def save(self, episode, learning_method='RL'):
        self.saver.save(self.sess, self._checkpoint_filename(episode, learning_method=learning_method, mode='save'))

    def load(self, learning_method='RL'):

        # if Config.EPISODE_NUMBER_TO_LOAD > 0:
        #     filename = self._checkpoint_filename(Config.EPISODE_NUMBER_TO_LOAD, mode='load', wandb_runid_for_loading=Config.LOAD_FROM_WANDB_RUN_ID)
        # else:
        #     filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0, mode='load', learning_method=learning_method, wandb_runid_for_loading=Config.LOAD_FROM_WANDB_RUN_ID)))

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints", 'RL_tmp/network_00000000')

        print("[NetworkVPCore] Loading checkpoint file:", filename)
        self.saver.restore(self.sess, filename)

        return self._get_episode_from_filename(filename)

    def train_with_regression(self, dataset_ped_train, dataset_ped_test):

        if Config.EPISODE_NUMBER_TO_LOAD > 0:
            filename = self._checkpoint_filename(Config.EPISODE_NUMBER_TO_LOAD, mode='load', wandb_runid_for_loading=Config.LOAD_FROM_WANDB_RUN_ID)
        else:
            filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0, mode='load', wandb_runid_for_loading=Config.LOAD_FROM_WANDB_RUN_ID)))
        self.saver.restore(self.sess, filename)

        return self._get_episode_from_filename(filename)

    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))

    def get_regression_loss(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
        return self.sess.run([self.cost_v, self.cost_p_regression_sum, self.cost_regression_sum], feed_dict=feed_dict)
