from ib.opt import ibConnection, message
from ib.ext.Contract import Contract
from time import sleep, strftime, localtime
from ib.ext.TickType import TickType as tt
import pdb

def error_handler(msg):
    print (msg)

new_symbolinput = ['CANFINHOM', 'KSCL']
newDataList = []
dataDownload = []

def my_callback_handler(msg):
	global newDataList
	#print msg.reqId, msg.date, msg.open, msg.high, msg.low, msg.close, msg.volume
	#pdb.set_trace();
	if ('finished' in str(msg.date)) == False:
		new_symbol = new_symbolinput[msg.reqId]
		#pdb.set_trace()
		dataStr = '%s, %s, %s, %s, %s, %s, %s, %s, %s' % (new_symbol, msg.date, msg.open, msg.high, msg.low, msg.close, msg.volume, msg.WAP, msg.count)
		newDataList += [dataStr]
	else:
		new_symbol = new_symbolinput[msg.reqId]
		filename = new_symbol + '.csv'
		csvfile = open('csv_data/'+ filename,'wb')
		newDataList.insert(0,'Script, DateTime, Open, High, Low, Close, Volume, WAP, Count')
		for item in newDataList:
			csvfile.write('%s \n' % item)
		csvfile.close()
		newDataList = []
		global dataDownload
		dataDownload.append(new_symbol)

tws = ibConnection()
tws.register(my_callback_handler, message.historicalData)
tws.connect()

symbol_id = 0
for i in new_symbolinput:
	print i
	c = Contract()
	c.m_symbol = i
	c.m_secType = "STK"
	c.m_exchange = "NSE"
	c.m_currency = "INR"
	endtime = strftime('%Y%m%d %H:%M:%S')
	#for reference http://www.inside-r.org/packages/cran/IBrokers/docs/reqHistoricalData
	#for data limitation
	tws.reqHistoricalData(symbol_id,contract=c,endDateTime=endtime,
            durationStr='1 Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=0,
            formatDate=2)
	symbol_id += 1
	sleep(1)

print dataDownload
print 'All done'

tws.disconnect()