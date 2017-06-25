

# Description

The purpose of this file is to keep track of why I am making changes, since I am working alone on this project. This file is also a journal for me to brain storm. 
Let's start...

### First Problem: Analyzing the Reward Function

The current problem with the project is that I can see deep_trader improving the reward function based on how average reward is increasing on test data. But that means nothing to me now because its very hard to see what decisions deep_trader is making from standardized data. 

Currently I am thinking of implementing code which actually shows the decisions per episode and as well as the real price. This will help me analyze episodes in a much better way. Let's get started on it and I don't feel version 0 of this will be difficult to implement. 

### Errors in the Reward System

The above idea actually helped and showed me that the way I have implemented the reward system its not correct. I am taking highest price of that particular time interval while buying and lowest while selling. In small interval like a minute its bound to make loss because of it. Even on big interval this will be bad. There are two solutions which coming to my mind:

1. Take average of opening, closing , highest and lowest price and use that price for that particular minute. If time interval is greater than minute take average of first minutes and then use those minutes average to take time interval average.
2. Make a second network which main task is just to select time when to execute trading action based on continous stream of time-price data trader is receving. This will be a good system to implement above deep_trader.

I will first go with first for now and mostly will implement 2 later above it. I think implementing both will be good idea.

__Data updates__: I also need to update standardization of data and reward system to use actual pricing rather than normalized price.

### Tweaking the DQN

Finally DQN algo is printing some result which I can observe, but it's not something I like. The average reward is coming negative on test data and if I ask the algorithm to train itself for multiple iteration, it starts to continue holding stocks. I guess its not seeing a lot of positive reinforcement. There are few ideas which I have: 

1. Making reward function so that terminal reward is only used.
2. USE PG instead of DQN as its online learning which looks more applicable here.
3. Change input data vector to have better understanding of markets at one time.

_Aside_

I can't help but think how much a project teaches me. In my personal life I don't like uncertainity like as a human we love to live in loop of repeatedness with certainity. But in project like this you are always in limbo unless you see the first ray of algo improving. I am waiting for that first singal here to kill anxiety. 

### DQN Improving

The good news is that dqn_model is showing signs of improvement. When I run dqn_model on test data 20 times, it makes a profit all 20 times. The average is 1 INR per episode but that is also good because I allow agent to trade only one quantity. When I run a random algo which just take action randomly it actually made a loss 19 times out of 20. This dqn_model has learnt something.

### Adding a PGN Model

I am trying PGN model now on the stocks. The major problem here is that the random algorithm is not getting positive rewards a lot of the time in the beginning. This causes the algo to hold everything, because this means zero reward (rather than take a negative reward). I have few idea to solve this:

1. Change algo to take a lot of random actions for a long time at training.
2. Find out which action lead to positive reinforcement in test data and train first a supervised network based on that. Use the network here then to train RL network.  I have started this on development branch of project.
3. I have started training network first on supervised learning. This is actually more exciting than i thought because I can't just lablel data by seeing next price on time interval. Entire episode price after a particular time interval matters as you can hold also. I have to first write a program which will take an episode price and then will give a list of actions to take to maximize profit generation. Lets go hunting.

### PG Improving:

The idea to use use supervised network first actually worked and now PG is also generating profit on the test data. The model is not just holding stocks its improving also.

### Checking profitablity:
1. I have checked different stock brokerage transaction cost in India. The minimum one can go is to pay just STT for brokerage tax which is 0.025% on an intra-day transaction. The rate is 0.017% on all Futures and Options transactions. Now, based on model, I have to write code to check how much friction costs can the strategy win past.
2. Separating test data into 4 smaller sets and checking average profit in them.