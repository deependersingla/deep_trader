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

def prepare_data():
    supervised_data = {}
    output_data = []
    new_data_list  = []
    files = glob.glob("/home/deep/development/deep_trading/ib/ftr_csv/*.csv")
    #files = ["/home/deep/development/deep_trading/ib/csv_data/KSCL.csv"]
    for file in files:
        stock_data = genfromtxt(file, delimiter=',', dtype=None, names=True)
        output = None
        daily_gain = None
        #check 1000 can be changed accordingly
        last_n_data = []
        intial_data = stock_data[0]
        for data in stock_data[1:]:
            final_value = data['Low']
            intial_value = intial_data['High']
            #can also add transaction fee
            if (final_value > intial_value):
                output = 1 #should have bought
            elif (final_value < intial_value):
                output = 2 #should have sold 
            else:
                output = 0 #should have done nothing
            #percent_gain = (final_value - intial_value) / intial_value
            #converting into 0 to 1
            #output = ((percent_gain - (-1)) / (1 - (-1)) ) * (1 - 0) + 0
            final_date = dateutil.parser.parse(data["DateTime"]).date()
            intial_date = dateutil.parser.parse(intial_data["DateTime"]).date()
            last_n_data.append([intial_data["Low"], intial_data["High"], intial_data["Close"], intial_data["Open"], intial_data["Volume"]])
            if final_date != intial_date:            
                opening_price = data["Open"]
                closing_price = intial_data["Close"]
                gain = (opening_price - closing_price) / closing_price
                #converting into 0 to 1 from -1 to 1
                #new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
                daily_gain = ((gain - (-1)) / (1 - (-1)) ) * (1 - 0) + 0
                #todo 1000 can change
            if (daily_gain is not None) and (len(last_n_data) >= 1000):
                last_n_data = last_n_data[-1000:]
                input_vector = make_supervised_vector(last_n_data, daily_gain)
                #list_data = [daily_gain,intial_data["Low"], intial_data["High"], intial_data["Close"], intial_data["Open"], intial_data["Volume"]]
                output_data.append(output)
                #if daily_gain > 0.5:
                    #pdb.set_trace()
                new_data_list.append(input_vector)
            intial_data = data
    stock_data = np.asarray(new_data_list)
    output_data = np.asarray(output_data)
    supervised_data["data"] = stock_data
    supervised_data["target"] = output_data
    return supervised_data

def make_supervised_vector(last_n_data, daily_gain):
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


def standardization(data):
    #Standard standardization with mean 
    return preprocessing.scale(data)


def find_average(data):
    return np.mean(data, axis=0)


def load_stock_data():
    if not os.path.exists('stock_5.pkl'):
        dictonary =  prepare_data()
        with open("stock_5.pkl", "wb") as myFile:
            six.moves.cPickle.dump(dictonary, myFile, -1)
    with open('stock_cont.pkl', 'rb') as myFile:
        data = six.moves.cPickle.load(myFile)
    #pdb.set_trace();
    return data