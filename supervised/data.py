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
#ticket data wrting to some csv or temp array

def prepare_data(scrip_id):
    stock_data = genfromtxt('/home/deep/development/deep_trading/ib/csv_data/' + scrip_id + '.csv', delimiter=',', dtype=None, names=True)
    output = None
    daily_gain = None
    supervised_data = {}
    output_data = []
    new_data_list  = []
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
        final_date = dateutil.parser.parse(data["DateTime"]).date()
        intial_date = dateutil.parser.parse(intial_data["DateTime"]).date()
        if final_date != intial_date:
            #pdb.set_trace()
            opening_price = data["Open"]
            closing_price = intial_data["Close"]
            gain = (opening_price - closing_price) / closing_price
            #converting into 0 to 1 from -1 to 1
            #new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
            daily_gain = ((gain - (-1)) / (1 - (-1)) ) * (1 - 0) + 0
        if daily_gain is not None:

            #TODO can use this place for standardization also
            list_data = [daily_gain,intial_data["Low"], intial_data["High"], intial_data["Close"], intial_data["Open"], intial_data["Volume"]]
            output_data.append(output)
            new_data_list.append(list_data)
        intial_data = data
    stock_data = np.asarray(new_data_list)
    output_data = np.asarray(output_data)
    supervised_data["data"] = stock_data
    supervised_data["target"] = output_data
    return supervised_data

def standardization(data):
    #Standard standardization with mean = 0 
    return preprocessing.scale(data)


def find_average(data):
    return np.mean(data, axis=0)


def load_stock_data():
    return prepare_data("CANFINHOM")