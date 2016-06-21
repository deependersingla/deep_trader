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

episode = 10 #lenght of one episode
data_array = []
parent_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
raw_data_file  = os.path.join(parent_dir,'data/sorted_data.csv') 
moving_average_number = 1000 #number of time interval for calculating moving average

def prepare_data():
	stock_data = genfromtxt(raw_data_file, delimiter=',', dtype=None, names=True)
	average_dataset = []
	total_data = []
	temp_episode = []
	index = 0
	for data in stock_data:
		temp = [data[2], data[3], data[4], data[5], data[6]]
		average_dataset.append(temp)
		print(index)
		if index > moving_average_number:
			mean = find_average(average_dataset)
			mean_array = average_dataset/mean
			last_one_hour_average = find_average(mean_array[-60:])
			last_one_day_average = find_average(mean_array[-300:])
			last_one_week_average = mean_array
			last_minute_data = average_dataset[-1]
			average_dataset = average_dataset[1:]
			vector = []
			vector.extend(last_minute_data)
			vector.extend(last_one_hour_average)
			vector.extend(last_one_day_average)
			vector.extend(last_one_week_average)
			temp_episode.append(vector)
			if index % 10 == 0:
				total_data.append(temp_episode)
				temp_episode = []
		index += 1
	pdb.set_trace();
	print(5)

def find_average(data):
    return np.mean(data, axis=0)

def make_standard_data(last_n_data, daily_gain):
    column = len(last_n_data[0])
    standaridized_data = standardization(last_n_data)
    #apply minmax above it to change range from 0 to 1, can be changed
    min_max_scaler = preprocessing.MinMaxScaler()
    standaridized_data = min_max_scaler.fit_transform(standaridized_data)
    vector = standaridized_data[-10:].reshape(10*column,)
    vector = np.append(vector,find_average(standaridized_data[-200:]))
    vector = np.append(vector, find_average(standaridized_data[-1000:]))
    vector = np.append(vector, daily_gain)
    return vector

prepare_data()