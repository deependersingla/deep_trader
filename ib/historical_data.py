from ib.opt import ibConnection, message
from ib.ext.Contract import Contract
from time import sleep, strftime
from ib.ext.TickType import TickType as tt
import pdb

def error_handler(msg):
    print (msg)


def my_callback_handler(msg):
	print(msg)

tws = ibConnection()
tws.register(my_callback_handler, message.historicalData)
tws.connect()

c = Contract()
c.m_symbol = "CANFINHOM"
c.m_secType = "STK"
c.m_exchange = "NSE"
c.m_currency = "INR"
endtime = strftime('%Y%m%d %H:%M:%S')
#for reference http://www.inside-r.org/packages/cran/IBrokers/docs/reqHistoricalData
tws.reqHistoricalData(tickerId=1,contract=c,endDateTime=endtime,
            durationStr='1 M',
            barSizeSetting='2 mins',
            whatToShow='TRADES',
            useRTH=0,
            formatDate=1)
sleep(10000000)

print 'All done'

tws.disconnect()