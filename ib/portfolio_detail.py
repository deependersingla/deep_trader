#===============================================================================
from ib.opt import ibConnection, message
from time import sleep
 
#===============================================================================
# Class IB_API
#===============================================================================
class IB_API:
	def __init__(self):
		self.connection = ibConnection()
		self.connection.registerAll(self.process_messages)
		self.connection.connect()

	def process_messages(self, msg):
		if msg.typeName == "updatePortfolio":
			print msg
	
	def get_account_updates(self):
		print "Calling Portfolio"
		self.connection.reqAccountUpdates(1, '')
		sleep(10)
 
if __name__ == '__main__':
	ib = IB_API()
	ib.get_account_updates()