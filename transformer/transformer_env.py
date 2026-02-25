"""
Sequence-aware Trading Environment for the Transformer agent.

Wraps the base StockTradingEnv logic but returns observations as sequences
of shape (seq_len, features) instead of single feature vectors.
The agent sees a sliding window of past market states at each step.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

# Allow imports from sibling packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class SequenceTradingEnv(gym.Env):
    """
    Trading environment that delivers observation sequences.

    At each step the observation is a (seq_len, feature_dim) array built
    from the last `seq_len` feature vectors.  Early in the episode the
    buffer is zero-padded on the left.

    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    Reward:  Δ portfolio value between steps.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        state_data: np.ndarray,
        price_data: np.ndarray,
        seq_len: int = 64,
        initial_balance: float = 10_000.0,
        episode_length: int = 1000,
    ):
        super().__init__()

        self.state_data = state_data
        self.price_data = price_data
        self.total_data_len = len(price_data)
        self.seq_len = seq_len
        self.initial_balance = initial_balance
        self.episode_length = episode_length
        self.feature_dim = state_data.shape[1]

        # Actions: Hold / Buy / Sell
        self.action_space = spaces.Discrete(3)

        # Observation: a sequence of feature vectors
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, self.feature_dim),
            dtype=np.float32,
        )

        self.reset()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = self.total_data_len - self.episode_length - 1
        if max_start <= 0:
            max_start = 1
        self.start_idx = random.randint(0, max_start)
        self.current_step_idx = 0

        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset_value = self.initial_balance
        self.prev_asset_value = self.initial_balance

        # Internal history buffer for building sequences
        self._history: list[np.ndarray] = []

        return self._get_obs(), {}

    def _current_global_idx(self) -> int:
        return self.start_idx + self.current_step_idx

    def _get_single_state(self) -> np.ndarray:
        """Return the raw feature vector for the current timestep."""
        return self.state_data[self._current_global_idx()].astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Build a (seq_len, feature_dim) observation from history.
        Zero-padded on the left when history is shorter than seq_len.
        """
        current = self._get_single_state()
        self._history.append(current)

        buf = self._history[-self.seq_len:]
        if len(buf) < self.seq_len:
            pad = [np.zeros(self.feature_dim, dtype=np.float32)] * (self.seq_len - len(buf))
            buf = pad + buf

        return np.array(buf, dtype=np.float32)

    def step(self, action: int):
        idx = self._current_global_idx()
        current_price = self.price_data[idx]

        # Execute action
        if action == 1:  # Buy
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2:  # Sell
            self.shares_held -= 1
            self.balance += current_price

        self.current_step_idx += 1
        terminated = self.current_step_idx >= self.episode_length
        truncated = False

        # Portfolio valuation
        current_asset_value = self.balance + self.shares_held * current_price
        reward = current_asset_value - self.prev_asset_value

        self.prev_asset_value = current_asset_value
        self.total_asset_value = current_asset_value

        info = {
            "balance": self.balance,
            "shares": self.shares_held,
            "total_value": self.total_asset_value,
            "price": current_price,
        }

        obs = self._get_obs() if not terminated else np.zeros(
            (self.seq_len, self.feature_dim), dtype=np.float32
        )

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        idx = self._current_global_idx()
        print(
            f"Step {self.current_step_idx} | "
            f"Price: {self.price_data[idx]:.5f} | "
            f"Value: {self.total_asset_value:.2f}"
        )
