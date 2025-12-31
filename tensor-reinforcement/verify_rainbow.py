import gymnasium as gym
import numpy as np
from rainbow_agent import RainbowAgent
import tensorflow as tf

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    print("Verifying Rainbow DQN on CartPole-v1...")
    
    try:
        env = gym.make('CartPole-v1')
    except Exception as e:
        print(f"Failed to load CartPole-v1: {e}")
        return

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)
    
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
    
    agent = RainbowAgent(state_dim, action_dim)
    
    # Run slightly faster for verification
    agent.batch_size = 32
    agent.memory.capacity = 1000
    
    episodes = 5
    
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
                
        print(f"Episode {e+1}/{episodes} - Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Beta: {agent.beta:.2f}")

    print("Verification Complete.")

if __name__ == "__main__":
    main()
