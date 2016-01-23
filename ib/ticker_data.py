from ib.opt import ibConnection, message
from ib.ext.Contract import Contract
from time import sleep
from ib.ext.TickType import TickType as tt
import pdb

def error_handler(msg):
    print (msg)


def my_callback_handler(msg):
	#for reference on this /home/deep/development/IbPy/ib/ext/TickType
	#another class method tt.getField(msg.field)
	if msg.field in [tt.BID,tt.ASK,tt.LAST,tt.HIGH,tt.LOW,tt.OPEN,tt.CLOSE]:
		print tt.getField(msg.field), msg.price 
	elif msg.field in [tt.BID_SIZE,tt.ASK_SIZE,tt.LAST_SIZE,tt.VOLUME,tt.AVG_VOLUME]:
		print tt.getField(msg.field), msg.size 
	elif msg.field in [tt,LAST_TIMESTAMP]:
		print tt.getField(msg.field), msg.time

tws = ibConnection()
tws.register(my_callback_handler, message.tickSize, message.tickPrice)
tws.connect()

c = Contract()
c.m_symbol = "KSCL"
c.m_secType = "STK"
c.m_exchange = "NSE"
c.m_currency = "INR"

tws.reqMktData(5,c,"",False)
sleep(10000000)

print 'All done'

tws.disconnect()