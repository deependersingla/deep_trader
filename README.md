# reinforcement-trading
This project uses Deep Q learning on stock market and agent tries to learn trading. The goal is to check if the agent can learn to read tape. The project is dedicated to hero in life great Jesse Livermore.


More info here:
https://docs.google.com/document/d/12TmodyT4vZBViEbWXkUIgRW_qmL1rTW00GxSMqYGNHU/edit


Data sources:
1) Nifty Data: https://drive.google.com/folderview?id=0B8e3dtbFwQWUZ1I5dklCMmE5M2M&ddrp=1%20%E2%81%A0%E2%81%A0%E2%81%A0%E2%81%A09:05%20PM%E2%81%A0%E2%81%A0%E2%81%A0%E2%81%A0%E2%81%A0


2) Nifty futures:http://www.4shared.com/folder/Fv9Jm0bS/NSE_Futures


3) Google finance: The package connects with Google Finance and downloads a spreadsheet from:http://www.google.com/finance/getprices?q=.DJI&x=INDEXDJX&i=60&p=10d&f=d,c,h,l,o,v with the date (intra-daily), closing price, high, low, open and volume.

You can adjust this to your own preferences by 'seeing' the code as:http://www.google.com/finance/getprices?q=TICKER&x=EXCHANGE&i=INTERVAL&p=PERIOD&f=d,c,h,l,o,v.

Where:

TICKER: is the unique ticker symbol

EXCHANGE: is where the security is listed on

Hint: to track these inputs, for instance for the Dow Jones Industrial Average, you search the security of interest at Google Finance and then you can find at the top: (INDEXDJX:.DJI) which obviously refers to (EXCHANGE:TICKER).
INTERVAL: defines the frequency (60 = 60 seconds)

PERIOD: is the historical data period (see also Google Finance), here 10d refers to the past 10 days (up to current time).


#Dependencies:
For this new project, I will use chainer. I have read a lot of good things about it on google forums. So shifting from theano to chainer. Also theano makes my life hard all the time :).
1) https://github.com/pfnet/chainer
#for ubuntu
http://docs.chainer.org/en/stable/install.html



