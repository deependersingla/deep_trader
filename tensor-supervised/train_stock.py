import gzip
import os

import numpy as np
import six
from six.moves.urllib import request

from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import numpy as np
import dateutil.parser
import pdb
import glob
import cPickle as pickle
import shelve
import six
import episodic_data
from six.moves.urllib import request

data = episodic_data.load_data("data.pkl",episode=10)
x_train, x_test = train_test_split(data, test_size=0.10, random_state=123)

def get_intial_data():
    data_dictionary = {}
    data_dictionary["input"] = len(x_train[0][0]) + 1 #here one is portfolio value
    data_dictionary["action"] = 3 #short, buy and hold
    data_dictionary["hidden_layer_1_size"] = 40
    data_dictionary["hidden_layer_2_size"] = 20 #will be using later
    data_dictionary["x_train"] = x_train
    data_dictionary["x_test"] = x_test
    return data_dictionary

def new_stage_data(action,portfolio,old_state,new_state,portfolio_value,done):
    old_portfolio_value = portfolio_value
    low_price = new_state[2]
    #buying
    if action == 1:
        old_price = old_state[1]
        portfolio_value -= old_price
        portfolio += 1
    #selling
    elif action == 2:
        old_price = old_state[2]
        portfolio_value += old_price
        portfolio -= 1
    elif action == 0:
        portfolio = portfolio
    reward = 0
    if new_state:
        new_state = new_state + [portfolio]
    if portfolio >= 0:
        low_price = new_state[2]
    else:
        low_price = new_state[1]
    if done:
        reward = (portfolio_value + portfolio * low_price) - 100
    if reward > 0:
        reward = 10*reward #increasing reward
    return new_state, reward, done, portfolio, portfolio_value