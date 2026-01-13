import os
import joblib
import sys
import numpy as np
import tensorflow as tf
from stock_env import StockTradingEnv
from rainbow_agent import RainbowAgent
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_pipeline import load_and_process_data, get_feature_columns

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/sorted_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_networks_rainbow')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def main():
    # 1. Load Data
    print("Loading Data via Common Pipeline...")
    try:
        # Use common pipeline
        df = load_and_process_data(DATA_PATH)
    except Exception as e:
        print(f"Data Loading Error: {e}")
        return
    
    # 2. Prepare Data for RL
    # RL Agent needs a state vector. 
    # We use all technical features + OHLCV
    feature_cols = get_feature_columns()
    print(f"Using Features: {feature_cols}")
    
    raw_features = df[feature_cols].values
    
    # Scale features globally for the agent
    scaler = StandardScaler()
    state_data = scaler.fit_transform(raw_features)
    
    # Save Scaler for Live Trading
    import joblib
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Price data for the environment simulation
    price_data = df['Close'].values
    
    full_state_data = state_data
    full_price_data = price_data
    
    # Split Data (80% Train, 20% Test)
    split_idx = int(len(full_state_data) * 0.8)
    
    train_states = full_state_data[:split_idx]
    train_prices = full_price_data[:split_idx]
    
    test_states = full_state_data[split_idx:]
    test_prices = full_price_data[split_idx:]
    
    print(f"Total processed samples: {len(full_state_data)}")
    print(f"Training Samples (Pool): {len(train_states)}")
    print(f"Testing Samples (Pool): {len(test_states)}")

    # 3. Initialize Environment and Agent
    EPISODE_LENGTH = 1000
    env = StockTradingEnv(train_states, train_prices, episode_length=EPISODE_LENGTH)
    
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)
    
    print(f"Initializing Rainbow Agent (State: {state_dim}, Action: {action_dim})...")
    agent = RainbowAgent(state_dim, action_dim)
    
    # 4. Training Loop
    EPISODES = 20 # Keep short for demo/initial verification
    
    print("Starting Training...")
    for e in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        
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
            
            # Train Network
            agent.replay()
            
            if done:
                # Update Target Network
                agent.update_target_model()
                print(f"Episode: {e+1}/{EPISODES}, Value: {info['total_value']:.2f}, Profit: {info['total_value'] - env.initial_balance:.2f}, Epsilon: {agent.epsilon:.2f}, Beta: {agent.beta:.2f}")
                break
        
        # Save Model periodically
        if (e + 1) % 5 == 0:
            agent.save(os.path.join(MODELS_DIR, f"rainbow_dqn_episode_{e+1}.h5"))

    print("Training Finished.")
    
    # 5. Evaluation (Test on 5 Random Episodes from Test Set)
    print("Starting Evaluation on Test Data (5 Episodes)...")
    test_env = StockTradingEnv(test_states, test_prices, episode_length=EPISODE_LENGTH)
    
    # Disable exploration
    agent.epsilon = 0.0
    
    for i in range(5):
        state, _ = test_env.reset()
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = test_env.step(action)
            state = next_state
            done = terminated or truncated
        
        print(f"Test Ep {i+1} - Profit: {info['total_value'] - test_env.initial_balance:.2f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
