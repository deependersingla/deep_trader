from ib.opt import ibConnection, message
from ib.ext.Contract import Contract
from time import sleep

def my_callback_handler(msg):
    inside_mkt_bid = ''
    inside_mkt_ask = ''

    if msg.field == 1:
        inside_mkt_bid = msg.price
        print 'bid', inside_mkt_bid
    elif msg.field == 2:
        inside_mkt_ask = msg.price
        print 'ask', inside_mkt_ask


tws = ibConnection()
tws.register(my_callback_handler, message.tickSize, message.tickPrice)
tws.connect()

c = Contract()
c.m_symbol = "KSCL"
c.m_secType = "STK"
c.m_exchange = "NSE"
c.m_currency = "INR"

tws.reqMktData(1,c,"",False)
sleep(25)

print 'All done'

tws.disconnect()