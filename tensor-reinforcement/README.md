# Steps to reproduce DQN
1. cd tensor-reinforcement
2. python prepare_data.py

I use "ib/csv_data/AJP.csv" as an input, feel free to use other
stock in ib/csv_data directory to prepare "data.pkl" and "data_dict.pkl"
files.
3. Create a directory saved_networks inside tensor_reinforcement for
saving networks.
4. python dqn_model.py
