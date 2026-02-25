import os
import numpy as np
import pandas as pd
import tensorflow as tf
from modern_data import StockDataLoader
from stock_env import StockTradingEnv
from modern_dqn import DQNAgent

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/EURUSD_M1_2012_2025.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_networks_modern')

# GPU Enforcement
print("\n" + "="*50)
print("Hardware Check")
print("="*50)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} device(s)")
        print(f"   Device Name: {gpus[0].name}")
    except RuntimeError as e:
        print(f"❌ GPU Error: {e}")
else:
    print("⚠️  WARNING: No GPU detected! Training will proceed on CPU.")
    print("   Please ensure CUDA and cuDNN are correctly installed.")
print("="*50 + "\n")

def evaluate(agent, env, episodes=5):
    """
    Evaluates the agent on the validation/test environment.
    Returns: Average Reward, Average Profit
    """
    total_rewards = []
    total_profits = []
    
    # Store original epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0 # Disable exploration
    
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
            
        total_rewards.append(episode_reward)
        total_profits.append(info['total_value'] - env.initial_balance)
        
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    return np.mean(total_rewards), np.mean(total_profits)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def main():
    # 1. Load Data
    print("Loading Data...")
    loader = StockDataLoader(DATA_PATH)
    try:
        loader.load_csv()
        loader.preprocess()
    except Exception as e:
        print(f"Data Loading Error: {e}")
        return
    
    full_state_data, full_price_data, full_dates = loader.get_data()
    
    # Strict Chronological Split
    # Train: 2012 - 2020
    # Val:   2021 - 2022
    # Test:  2023 - 2025
    
    print("\nSplitting Data Chronologically...")
    
    train_mask = (full_dates < np.datetime64('2021-01-01'))
    val_mask = (full_dates >= np.datetime64('2021-01-01')) & (full_dates < np.datetime64('2023-01-01'))
    test_mask = (full_dates >= np.datetime64('2023-01-01'))
    
    train_states = full_state_data[train_mask]
    train_prices = full_price_data[train_mask]
    
    val_states = full_state_data[val_mask]
    val_prices = full_price_data[val_mask]
    
    test_states = full_state_data[test_mask]
    test_prices = full_price_data[test_mask]
    
    # Verification
    print(f"Train Set: {len(train_states)} samples ({pd.to_datetime(full_dates[train_mask][0])} -> {pd.to_datetime(full_dates[train_mask][-1])})")
    print(f"Val Set:   {len(val_states)} samples ({pd.to_datetime(full_dates[val_mask][0])} -> {pd.to_datetime(full_dates[val_mask][-1])})")
    print(f"Test Set:  {len(test_states)} samples ({pd.to_datetime(full_dates[test_mask][0])} -> {pd.to_datetime(full_dates[test_mask][-1])})")
    
    if len(train_states) == 0 or len(val_states) == 0 or len(test_states) == 0:
        print("Error: One of the splits is empty! Check date ranges.")
        return

    # 2. Initialize Environment and Agent
    # We use 1000 steps per episode (~2-3 days of minutes).
    EPISODE_LENGTH = 1000
    env = StockTradingEnv(train_states, train_prices, episode_length=EPISODE_LENGTH)
    val_env = StockTradingEnv(val_states, val_prices, episode_length=EPISODE_LENGTH) # Validation Env
    
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)
    agent = DQNAgent(state_dim, action_dim)
    
    # 3. Training Loop
    EPISODES = 200 # Increased scale
    TRAIN_FREQ = 4 # Train every 4 steps for stability and speed
    
    # CSV Logging
    log_file = "training_results.csv"
    with open(log_file, "w") as f:
        f.write("Episode,Train_Reward,Val_Reward,Val_Profit,Epsilon\n")
    
    print(f"Starting Training... Logging to {log_file}")
    
    for e in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Action
            action = agent.act(state)
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Remember
            agent.remember(state, action, reward, next_state, done)
            
            # Update State
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train Network (not every step)
            if steps % TRAIN_FREQ == 0:
                agent.replay()
            
            if done:
                # Update Target Network
                agent.update_target_model()
                # Decay Epsilon per Episode
                agent.decay_epsilon()
                
                print(f"Episode: {e+1}/{EPISODES}, Reward: {total_reward:.2f}, Profit: {info['total_value'] - env.initial_balance:.2f}, Epsilon: {agent.epsilon:.2f}")
                
                # Validation & Logging (every 10 episodes)
                if (e + 1) % 10 == 0:
                    val_reward, val_profit = evaluate(agent, val_env, episodes=5)
                    print(f"   🔎 Validation -> Avg Reward: {val_reward:.2f}, Avg Profit: {val_profit:.2f}")
                    
                    with open(log_file, "a") as f:
                        f.write(f"{e+1},{total_reward},{val_reward},{val_profit},{agent.epsilon}\n")
                        
                    # Save Model
                    agent.save(os.path.join(MODELS_DIR, f"dqn_episode_{e+1}.weights.h5"))
                break

    print("Training Finished.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
