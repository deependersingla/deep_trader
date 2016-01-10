import copy

import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F

class DQN_class:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    temporal_window = 10
    experience_size = 30000
    start_learn_threshold = 5000
    learning_steps_total = 500000
    learning_steps_burning = 10000
    epsilon_min = 0.00
    epsilon_test_time = 0.00

    def __init__(self, num_inputs, num_actions):
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        print "Initializing DQN..."
          #	Initialization of Chainer 1.1.0 or older.
    
   