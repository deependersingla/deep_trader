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
supervised_y_data  = episodic_data.make_supervised_data(data, data_dict)
x_train, x_test, y_train, y_test = train_test_split(data, supervised_y_data, test_size=0.10, random_state=123)
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
    data_dictionary["y_test"] = y_test
    data_dictionary["y_train"] = y_train
    return data_dictionary


def new_stage_data(action, portfolio, old_state, new_state, portfolio_value, done, episode_data):
    old_portfolio_value = portfolio_value
    #low_price = new_state[2]
    #changing code to use average price rather than normalized price
    price = episodic_data.data_average_price(data_dict, episode_data)
    next_price = episodic_data.data_average_price(data_dict, new_state)
    #price = data_dict[episodic_data.list_md5_string_value(episode_data)][-1]
    #next_price = data_dict[episodic_data.list_md5_string_value(new_state)][-1]
    #buying
    if action == 1:
        #old_price = old_state[1]
        #Todo: Add transaction cost here also 
        portfolio_value -= price
        portfolio += 1
    #selling
    elif action == 2:
        #old_price = old_state[2]
         #Todo: Add transaction cost here also 
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
    if reward > 0:
        reward = 2*reward #increasing reward
    #pdb.set_trace();
    return new_state, reward, done, portfolio, portfolio_value

def show_trader_path(actions, episode_data, portfolio_list, portfolio_value_list, reward_list):
    i = 0
    #print("Action, Average Price, Portfolio, Portfolio Value, Reward")
    for index, action in enumerate(actions):
        episode = episode_data[index]
        action_name = action_map[actions[index]]
        price = episodic_data.data_average_price(data_dict, episode)
        portfolio = portfolio_list[index]
        portfolio_value = portfolio_value_list[index]
        i += 1
        reward = reward_list[index]
        #print(action_name, price, portfolio, portfolio_value, reward)
    #print("last price:")
    episode = episode_data[i]
    last_price = episodic_data.data_average_price(data_dict, episode)
    #print(last_price)
    reward = (portfolio_value_list[-1] + portfolio_list[-1]*last_price)
    return reward 