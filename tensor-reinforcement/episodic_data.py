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
from six.moves.urllib import request
import hashlib
import json
from supervised_helper import generate_actions_from_price_data


episode = 10 #lenght of one episode
data_array = []
parent_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
raw_data_file  = os.path.join(parent_dir,'tensor-reinforcement/NIFTY50.csv') 
moving_average_number = 1000 #number of time interval for calculating moving average
#pdb.set_trace()

def prepare_data():
	stock_data = genfromtxt(raw_data_file, delimiter=',', dtype=None, names=True)
	average_dataset = []
	total_data = []
	temp_episode = []
	data_dict = {}
	index = 0
	for data in stock_data:
		temp = [data[2], data[3], data[4], data[5],data[8]]
		average_dataset.append(temp)
		print(index)
		print(len(average_dataset))
		if index > moving_average_number:
			mean = find_average(average_dataset)
			mean_array = average_dataset/mean
			last_one_hour_average = find_average(mean_array[-60:])
			last_one_day_average = find_average(mean_array[-300:])
			last_3_day_average = find_average(mean_array[-900:]) #this might change
			last_minute_data = mean_array[-1]
			average_dataset = average_dataset[1:]
			vector = []
			vector.extend(last_minute_data)
			vector.extend(last_one_hour_average)
			vector.extend(last_one_day_average)
			vector.extend(last_3_day_average)
			average_price = sum(temp[0:-2]) / float(len(temp[0:-2]))
			dict_vector = temp + [average_price]
			data_dict[list_md5_string_value(vector)] = dict_vector
			total_data.append(vector)
		index += 1
	with open("data.pkl", "wb") as myFile:
		six.moves.cPickle.dump(total_data, myFile, -1)
	print("Done")
	with open("data_dict.pkl","wb") as myFile:
		six.moves.cPickle.dump(data_dict, myFile, -1)

def find_average(data):
    return np.mean(data, axis=0)

def load_data(file,episode):
	data = load_file_data(file)
	return map(list,zip(*[iter(data)]*episode))

def load_file_data(file):
	with open(file, 'rb') as myFile:
		data = six.moves.cPickle.load(myFile)
	return data

def list_md5_string_value(list):
	string = json.dumps(list)
	return hashlib.md5(string).hexdigest()

def episode_supervised_data(data, data_dict):
	prices = []
	for iteration in data:
		prices.append(data_average_price(data_dict, iteration))
	actions = generate_actions_from_price_data(prices)
	return actions

def data_average_price(data_dict, data):
	data =  data_dict[list_md5_string_value(data)]
	return data[-1]


def make_supervised_data(data, data_dict):
	supervised_data = []
	if not os.path.exists('supervised_data.pkl'):
		for episode in data:
			supervised_data.append(episode_supervised_data(episode, data_dict))
		with open("supervised_data.pkl", "wb") as myFile:
			six.moves.cPickle.dump(supervised_data, myFile, -1)
	with open('supervised_data.pkl', 'rb') as myFile:
		supervised_data = six.moves.cPickle.load(myFile)
	return supervised_data
#prepare_data()