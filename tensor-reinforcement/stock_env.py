import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class StockTradingEnv(gym.Env):
    """
    A custom Stock Trading Environment that follows gym interface.
    This version samples random episodes from the dataset.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, state_data, price_data, initial_balance=10000, episode_length=1000, commission=0.0001):
        super(StockTradingEnv, self).__init__()
        
        self.state_data = state_data
        self.price_data = price_data
        self.total_data_len = len(price_data)
        self.initial_balance = initial_balance
        self.episode_length = episode_length
        self.commission = commission
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation: The feature vector provided by loader
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(state_data.shape[1],), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a random start index
        # Ensure we have enough data for the episode
        max_start = self.total_data_len - self.episode_length - 1
        self.start_idx = random.randint(0, max_start)
        self.current_step_idx = 0
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset_value = self.initial_balance
        self.prev_asset_value = self.initial_balance
        
        return self._get_obs(), {}

    def _current_global_idx(self):
        return self.start_idx + self.current_step_idx

    def _get_obs(self):
        idx = self._current_global_idx()
        return self.state_data[idx].astype(np.float32)

    def step(self, action):
        idx = self._current_global_idx()
        current_price = self.price_data[idx]
        
        # Execute Action with Commission
        if action == 1: # Buy
            self.shares_held += 1
            self.balance -= current_price * (1 + self.commission)
        elif action == 2: # Sell
            self.shares_held -= 1
            self.balance += current_price * (1 - self.commission)
            
        # Move to next step
        self.current_step_idx += 1
        
        # Check Termination
        terminated = self.current_step_idx >= self.episode_length
        truncated = False
        
        next_idx = self._current_global_idx()
        # Handle lookup for Reward calculation
        # If terminated, next_price is same as current (or we could peek, but let's keep it clean)
        if terminated:
            next_price = current_price 
        else:
            next_price = self.price_data[next_idx]
        
        # Calculate Reward
        current_asset_value = self.balance + (self.shares_held * current_price)
        
        # Reward = Profit/Loss for this step
        reward = current_asset_value - self.prev_asset_value
        
        self.prev_asset_value = current_asset_value
        self.total_asset_value = current_asset_value
        
        info = {
            'balance': self.balance,
            'shares': self.shares_held,
            'total_value': self.total_asset_value,
            'price': current_price
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Index: {self._current_global_idx()}, Price: {self.price_data[self._current_global_idx()]:.2f}, "
              f"Value: {self.total_asset_value:.2f}")
