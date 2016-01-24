## this file take input from ib and then make vector and run the experiment
##it takes care of filtering epsilon etx

from q_learning.dqn_agent import *
from prepare_data import *

class portfolio_agent():

	def __init__(self, data):
		self.porfolio = 0
		self.total_portfolio_delta = 0
		self.stock_data = data[0:1000]
		#for training no need for real data
		self.other_stored_data = data[1000:]
		#todo last action, not sure if required
		self.AGENT = Agent([0])
		self.last_buying_price = 0
		self.last_selling_price = 0

	def agent_start(self):
		stock_data = self.stock_data
		action = AGENT.start(make_input_vector())
        reward = find_reward(action)
        epoch = 0
        n_epoch = 10000
        while (epoch < n_epoch):
        	self.stock_data = np.vstack([self.stock_data, self.other_stored_data[0]])
        	self.other_stored_data = np.delete(self.other_stored_data, [0],0)
            self.stock_data = np.delete(self.stock_data, [0], 0)
        	#we can put conditional here to start taking from API one csv_data finished
        	action = AGENT.agent_step(reward, make_input_vector())
        	print("epoch is: ", epoch)
        	print("action is: ", action)
        	reward = find_reward(action)
        	epoch += 1

        print("porfolio: ", self.porfolio)
        print("total_portfolio_delta: ", self.total_portfolio_delta)


	def make_input_vector(self):
		#todo need to write normalization method also
		column = self.stock_data.shape[1]
		standaridized_data = standardization(self.stock_data)
		last_10_value = standaridized_data[-10:].reshape(10*column,)
		last_200_value_average = find_average(standaridized_data[-200:])
		last_1000_value_average = find_average(standaridized_data[-1000:])
		portfolio_value = self.portfolio
		#for makig sure that network understand it can't go more negative
		if (portfolio_value > 1):
			portfolio_value = 1
		elif (portfolio_value < -1):
			portfolio_value = -1
		return last_10_value + last_200_value_average + last_1000_value_average + portfolio_value
	

	def find_reward(self, action):
		#Todo fuction should get execution price for IB API
		#Todo think more deeply on reward
		#leaving short for now as it is making things complicated and also now allowed in market
		last_price = self.stock_data[-2][-2]
		current_price = self.stock_data[-1][-2]
		flag = True
		cost = 0
		if (action == 1) 
			transaction_cost = last_price * 0.001
			self.last_buying_price = last_price
			#punish action not to buy more if already bought
			if self.porfolio == 1:
				flag = False
		elif (action == -1):
			transaction_cost = last_price * 0.01
			self.last_selling_price = last_price
			#punish if selling more what is already sold
			if self.porfolio == 0:
				flag = False
			else:
				cost = (last_price - self.last_buying_price)
		else:
			transaction_cost = 0

		if flag:
			portfolio_change = self.porfolio * (current_price - last_price)
		else:
			portfolio_change = -10000
		
		#caluclate total profit and loss also
		self.total_portfolio_delta = transaction_cost + cost
		#todo need to take care that portfolio don't go much negative
		self.porfolio += action 
		return transaction_cost + portfolio_change