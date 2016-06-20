from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import numpy as np
import dateutil.parse
#ticket data wrting to some csv or temp array

def prepare_data(scrip_id):
	stock_data = genfromtxt('ib/csv_data/' + scrip_id + '.csv', delimiter=',', skip_header=1, dtype=None, names=True)
	stock_data = np.delete(stock_data, [0,1],1)
	return stock_data

def standardization(data):
    #Standard standardization with mean = 0 
    return preprocessing.scale(data)


def find_average(data):
    return np.mean(data, axis=0)