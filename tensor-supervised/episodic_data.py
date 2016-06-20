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

def prepare_data():
	stock_data = genfromtxt(raw_data_file, delimiter=',', dtype=None, names=True)
	average_dataset = stock_data[0:1000]
	for data in stock_data[1000:]:

		#this data
		#last 10 data
		#one week data
		#take this vector and store this
		#array can be split into episode
	print(5)

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