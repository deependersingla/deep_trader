from ib.ticker_data import *
from numpy import genfromtxt

#ticket data wrting to some csv or temp array

def prepare_data(scrip_id):
	stock_data = genfromtxt('ib/csv_data/' + scrip_id + '.csv', delimiter=',',skip_header=1)
	stock_data = np.delete(stock_data, [0,1],1)
	return stock_data