import os
import numpy as np
import tensorflow as tf
from modern_data import StockDataLoader
from stock_env import StockTradingEnv
from modern_pg import PGAgent

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/EURUSD_M1_2012_2025.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_networks_modern_pg')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def main():
    print("Loading Data for PG...")
    loader = StockDataLoader(DATA_PATH)
    try:
        loader.load_csv()
        loader.preprocess()
    except Exception as e:
        print(f"Data Loading Error: {e}")
        return
    
    full_state_data, full_price_data = loader.get_data()
    split_idx = int(len(full_state_data) * 0.8)
    train_states = full_state_data[:split_idx]
    train_prices = full_price_data[:split_idx]
    
    # Environment
    EPISODE_LENGTH = 1000
    env = StockTradingEnv(train_states, train_prices, episode_length=EPISODE_LENGTH)
    
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)
    
    # Agent
    agent = PGAgent(state_dim, action_dim)
    
    EPISODES = 20
    
    print("Starting PG Training...")
    for e in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward)
            
            state = next_state
            total_reward += reward
            
            if done:
                # Update at end of episode (Monte Carlo)
                loss = agent.train()
                print(f"Episode: {e+1}/{EPISODES}, Value: {info['total_value']:.2f}, Profit: {info['total_value'] - env.initial_balance:.2f}, Loss: {loss:.4f}")
        
        if (e + 1) % 5 == 0:
            agent.model.save_weights(os.path.join(MODELS_DIR, f"pg_episode_{e+1}.weights.h5"))

    print("PG Training Finished.")

if __name__ == "__main__":
    main()
