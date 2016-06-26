from ib.opt import ibConnection, message
from ib.ext.Contract import Contract
from time import sleep, strftime, localtime
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ib.ext.TickType import TickType as tt
import pdb

def error_handler(msg):
    print (msg)

new_symbolinput = ['CANFINHOM', 'KSCL', 'AJP', 'GRUH', 'GREENPLY', 'GRANULES', 'SBIN', 'SHILPAMED', 'SHEMAROO', 'TCS', 'TITAN', 'TORNTPHAR', 'TORNTPOWE', 'SHARONBIO', 'MANAPPURA', 'MAYURUNIQ', 'MPSLTD', 'MUTHOOTFI', 'ATULAUTO', 'AVANTIFEE']
#for symbol
new_symbolinput = ['GRUH', 'GREENPLY', 'GRANULES', 'SBIN', 'SHILPAMED', 'SHEMAROO', 'TCS', 'TITAN', 'TORNTPHAR', 'TORNTPOWE', 'SHARONBIO', 'MANAPPURA', 'MAYURUNIQ', 'MPSLTD', 'MUTHOOTFI', 'ATULAUTO', 'AVANTIFEE']
new_symbolinput = ['NIFTY50']
newDataList = []
dataDownload = []

def my_callback_handler(msg):
	global newDataList
	#print msg.reqId, msg.date, msg.open, msg.high, msg.low, msg.close, msg.volume
	#pdb.set_trace();
	if ('finished' in str(msg.date)) == False:
		new_symbol = new_symbolinput[msg.reqId]
		#pdb.set_trace()
		dataStr = '%s, %s, %s, %s, %s, %s, %s, %s, %s' % (new_symbol, strftime("%Y-%m-%d %H:%M:%S", localtime(int(msg.date))), msg.open, msg.high, msg.low, msg.close, msg.volume, msg.WAP, msg.count)
		newDataList += [dataStr]
	# else:
	# 	new_symbol = new_symbolinput[msg.reqId]
	# 	filename = new_symbol + '.csv'
	# 	csvfile = open('csv_data/'+ filename,'a+')
	# 	#newDataList.insert(0,'Script, DateTime, Open, High, Low, Close, Volume, WAP, Count')
	# 	for item in newDataList:
	# 		csvfile.write('%s \n' % item)
	# 	csvfile.close()
	# 	newDataList = []
	# 	global dataDownload
	# 	dataDownload.append(new_symbol)
def write_to_csv(sym):
	filename = sym + '.csv'
	csvfile = open('20_ftr_csv/'+ filename,'wb')
	s = []
	for i in newDataList:
		if i not in s:
			s.append(i)
	s.insert(0,'Script, DateTime, Open, High, Low, Close, Volume, WAP, Count')
	for item in s:
		csvfile.write('%s \n' % item)
	csvfile.close()


tws = ibConnection()
tws.register(my_callback_handler, message.historicalData)
tws.connect()

symbol_id = 0
for i in new_symbolinput:
	print i
	c = Contract()
	c.m_symbol = i
	c.m_secType = "IND"
	c.m_exchange = "NSE"
	c.m_currency = "INR"
	number_of_days = 2600
	for day in range(0,number_of_days,2):
		time = datetime.now() - relativedelta(days=(2600 - day))
		endtime = time.strftime('%Y%m%d %H:%M:%S')
		#for reference http://www.inside-r.org/packages/cran/IBrokers/docs/reqHistoricalData
		#for data limitation https://www.interactivebrokers.com/en/software/api/apiguide/tables/historical_data_limitations.htm
		tws.reqHistoricalData(symbol_id,contract=c,endDateTime=endtime,
            durationStr='2 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=1,
            formatDate=2)
	    #IB blocks more than 60 request per 10 minute
		sleep(15)
	write_to_csv(i)
	newDataList = []

	symbol_id += 1

print dataDownload
print 'All done'

tws.disconnect()