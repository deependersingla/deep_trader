#Reinforcement-trading
This project uses Deep Q learning on stock market and agent tries to learn trading. The goal is to check if the agent can learn to read tape. The project is dedicated to hero in life great Jesse Livermore.


Process:
a) Intially I started by using chainer for the project for both supervised and reinforcement learning. In middle of it AlphaGo (https://research.googleblog.com/2016/01/alphago-mastering-ancient-game-of-go.html) came, so I shifted to read Sutton book (https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html), AlphaGo paper and related paper, David Silver lectures (http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html, they are great)
on Reinforcement learning. I Coming back to project after 45 days a lot has changed in TensorFlow all the cool kids (even Deepmind) have started using it. Hence, I am ditching chainer and will use tensorflow from now. Exciting times ahead.


#For the first iteration of the project
I will go with episodic version,one reason which made me choose that:
a) I will not have to calculate reward after every action which agent will make, I can just make terminal reward based on portfolio value after episode - inital value of portfolio - transaction cost occur inside the episode. The reason for doing it that I believe it will motivate agent to learn trading rather than to think long term. This also means that I have to check the hypothesis on different episode of different length. 
I also have to test thypothesis that what will if i give immediate reward after every time interval and also terminal reward based on what I discussed above. All in all the project looks like a lot of hit and trial. I should better write good code and store all results properly so that I can compare them to see what works and what don't. Ofcourse the idea is to make sure agent remain profitable while trading. 

#Policy network:
I will be starting with simple feed-forward network. Though, I am also inclined to use convolutional network reason being they do very well when the minor change in input should not make a change in ouput. for example: in image recognizition, a small pixel values here and there doesn't make image change. Intutively stocks numbers to me somtime look same a small change should not trigger a trade but again the problem here comes with normalization. With normalization the big change in number will be reduced to a very small in inputs hence its good to start with feed-forward.


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
1) https://github.com/tensorflow/tensorflow
2) https://github.com/pfnet/chainer (if wants to use chainer)
3) https://github.com/blampe/IbPy

External help
1)
2) Deep-Q-chainer
https://github.com/ugo-nama-kun/DQN-chainer


#for reading on getting data using IB
https://www.interactivebrokers.com/en/software/api/apiguide/tables/historical_data_limitations.htm

#Reinforcement learning resources
https://github.com/aikorea/awesome-rl



