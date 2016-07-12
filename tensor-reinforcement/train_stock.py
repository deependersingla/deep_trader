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
data_dict = episodic_data.load_file_data("data_dict.pkl")
x_train, x_test = train_test_split(data, test_size=0.10, random_state=123)
action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
#calling data_dict is data_dict[episodic_data.list_md5_string_value(list)]

def get_intial_data():
    data_dictionary = {}
    data_dictionary["input"] = len(x_train[0][0]) + 1 #here one is portfolio value
    data_dictionary["action"] = 3 #short, buy and hold
    data_dictionary["hidden_layer_1_size"] = 40
    data_dictionary["hidden_layer_2_size"] = 20 #will be using later
    data_dictionary["x_train"] = x_train
    data_dictionary["x_test"] = x_test
    return data_dictionary

def new_stage_data(action, portfolio, old_state, new_state, portfolio_value, done, episode_data):
    old_portfolio_value = portfolio_value
    #low_price = new_state[2]
    #changing code to use average price rather than normalized price
    price = data_dict[episodic_data.list_md5_string_value(episode_data)][-1]
    next_price = data_dict[episodic_data.list_md5_string_value(new_state)][-1]
    #buying
    if action == 1:
        #old_price = old_state[1]
        portfolio_value -= price
        portfolio += 1
    #selling
    elif action == 2:
        #old_price = old_state[2]
        portfolio_value += price
        portfolio -= 1
    elif action == 0:
        portfolio = portfolio
    #reward = 0
    #if new_state:
    new_state = new_state + [portfolio]
    #if portfolio >= 0:
        #low_price = new_state[2]
    #else:
        #low_price = new_state[1]
    #reward system might need to change and require some good thinking
    #if done:
    reward = (portfolio_value + portfolio * next_price)
    #if reward > 0:
    #    reward = 2*reward #increasing reward
    #pdb.set_trace();
    return new_state, reward, done, portfolio, portfolio_value

def show_trader_path(actions, episode_data):
    for index, action in enumerate(actions):
        episode = episode_data[index]
        action_name = action_map[actions[index]]
        data =  data_dict[episodic_data.list_md5_string_value(episode)]
        # if action == 2:
        #     price = data[2]
        # elif action == 1:
        #     price = data[1]
        # elif action == 0:
        #     price = data[1]
        #print(data)
        price = data[-1]
        print(action_name,price)