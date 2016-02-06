# -*- coding: utf-8 -*-
"""
Major help from Deep Q-network implementation with chainer by Naoto Yoshida https://github.com/ugo-nama-kun/DQN-chainer
"""
import copy

import pickle
import numpy as np
import scipy.misc as spm
import pdb

from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F


class DQN_class:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 10**3  # Initial exploratoin. original
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**2  # Target update frequancy. original
    data_size = 10**5  # Data size of history. original
     
    #actions are 0 => do nothing, 1 -> buy, -1 sell
    def __init__(self, input_vector_length,enable_controller=[0, 1, 2]):
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller  # Default setting : "Pong"
        self.input_vector_length = input_vector_length

        print "Initializing DQN..."
#   Initialization for Chainer 1.1.0 or older.
#        print "CUDA init"
#        cuda.init()
        
        #inputs --> 5 * 14 (with 10 temporality) + 5 (of last one hour) + 5 (of last 24 hour)
        print "Model Building"
        self.model = FunctionSet(
            l1=F.Linear(input_vector_length, 500),
            l2=F.Linear(500, 250),
            l3=F.Linear(250, 80),
            q_value=F.Linear(80, self.num_of_actions,
                             initialW=np.zeros((self.num_of_actions, 80),
                                               dtype=np.float32))
        ).to_gpu()

        print "Initizlizing Optimizer"
        self.optimizer = optimizers.RMSpropGraves(lr=0.0002, alpha=0.3, momentum=0.2)
        self.optimizer.setup(self.model.collect_parameters())

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.D = [np.zeros((self.data_size, self.input_vector_length), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.input_vector_length), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

    def forward(self, state, action, Reward, state_dash, episode_end):
        num_of_batch = state.shape[0]
        s = Variable(state)
        s_dash = Variable(state_dash)

        Q = self.Q_func(s)  # Get Q-value

        # Generate Target Signals
        max_Q_dash_ = self.Q_func(s_dash)
        tmp = list(map(np.max, max_Q_dash_.data.get()))
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(Q.data.get(), dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = np.sign(Reward[i]) + self.gamma * max_Q_dash[i]
            else:
                tmp_ = np.sign(Reward[i])
            target[i, self.action_to_index(action[i])] = tmp_

        loss = F.mean_squared_error(Variable(cuda.to_gpu(target)), Q)
        return loss, Q

    def stockExperience(self, time,
                        state, action, reward, state_dash,
                        episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = reward
        else:
            self.D[0][data_index] = state
            self.D[1][data_index] = action
            self.D[2][data_index] = reward
            self.D[3][data_index] = state_dash
        self.D[4][data_index] = episode_end_flag

    def experienceReplay(self, time):

        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, self.input_vector_length), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.input_vector_length), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.D[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.D[1][replay_index[i]]
                r_replay[i] = self.D[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.D[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.D[4][replay_index[i]]

            s_replay = cuda.to_gpu(s_replay)
            s_dash_replay = cuda.to_gpu(s_dash_replay)

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.forward(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()

    def Q_func(self, state):
        #todo might want to normalize input, but for now I will do that outside this class 
        h1 = F.relu(self.model.l1(state))  
        h2 = F.relu(self.model.l2(h1))
        h3 = F.relu(self.model.l3(h2))
        Q = self.model.q_value(h3)
        return Q

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        Q = self.Q_func(s)
        Q = Q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            print "RANDOM"
        else:
            index_action = np.argmax(Q.get())
            print "GREEDY"

        return self.index_to_action(index_action), Q
    
    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)
    
    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)