"""
Transformer-based Double DQN Agent with Prioritized Experience Replay.

Key difference from the MLP agents: this agent maintains a **sliding window**
of the last `seq_len` states and stores *sequences* in the replay buffer,
giving the Transformer temporal context for each decision.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque
import random

# Allow imports from sibling packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer.transformer_model import build_transformer_dqn

# tensor-reinforcement uses a hyphen → not a valid Python package name.
# We add its directory to sys.path and import directly.
_tr_dir = os.path.join(os.path.dirname(__file__), '..', 'tensor-reinforcement')
if _tr_dir not in sys.path:
    sys.path.insert(0, _tr_dir)
from per_buffer import PrioritizedReplayBuffer


class TransformerDQNAgent:
    """
    Double DQN agent backed by a Transformer Q-network.

    - Maintains a sliding window of `seq_len` past states.
    - Uses Prioritized Experience Replay for sample efficiency.
    - Target network updated periodically for training stability.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        seq_len: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dff: int = 128,
        dropout_rate: float = 0.1,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 32,
        memory_size: int = 50_000,
        target_update_freq: int = 500,
    ):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.seq_len = seq_len

        # RL hyper-parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq

        # Prioritized Replay
        self.memory = PrioritizedReplayBuffer(capacity=memory_size)
        self.beta = 0.4           # IS weight exponent – anneals toward 1
        self.beta_increment = 0.001

        # Models
        model_kwargs = dict(
            feature_dim=feature_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dff=dff,
            dropout_rate=dropout_rate,
        )
        self.model = build_transformer_dqn(**model_kwargs)
        self.target_model = build_transformer_dqn(**model_kwargs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )
        self.update_target_model()

        # Sliding window for current episode
        self._state_buffer: deque = deque(maxlen=seq_len)
        self.steps = 0

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------
    def reset_sequence(self):
        """Call at the start of each episode to clear the state buffer."""
        self._state_buffer.clear()

    def _build_sequence(self, state: np.ndarray) -> np.ndarray:
        """
        Returns a (seq_len, feature_dim) array by left-padding with zeros
        if the buffer doesn't have enough history yet.
        """
        buf = list(self._state_buffer) + [state]
        # Pad or truncate to seq_len
        if len(buf) < self.seq_len:
            pad = [np.zeros(self.feature_dim, dtype=np.float32)] * (self.seq_len - len(buf))
            buf = pad + buf
        else:
            buf = buf[-self.seq_len:]
        return np.array(buf, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core API (compatible interface with DQNAgent / RainbowAgent)
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        self.steps += 1
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        seq = self._build_sequence(state)
        q_values = self.model.predict(seq[np.newaxis], verbose=0)
        return int(np.argmax(q_values[0]))

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """
        Store a transition.  We store the *sequences* (current + next)
        rather than individual states so replay can feed the Transformer.
        """
        seq_curr = self._build_sequence(state)
        # Push current state into the buffer so the next call sees it
        self._state_buffer.append(state)
        seq_next = self._build_sequence(next_state)

        self.memory.add(seq_curr, action, reward, seq_next, done)

    def replay(self):
        """Sample from PER, compute Double-DQN targets, train, update priorities."""
        if len(self.memory) < self.batch_size:
            return

        minibatch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)

        # Unpack
        states = np.array([t[0] for t in minibatch], dtype=np.float32)
        actions = np.array([t[1] for t in minibatch], dtype=np.int32)
        rewards = np.array([t[2] for t in minibatch], dtype=np.float32)
        next_states = np.array([t[3] for t in minibatch], dtype=np.float32)
        dones = np.array([t[4] for t in minibatch], dtype=np.float32)

        # Double DQN: select actions with online model, evaluate with target
        next_q_online = self.model.predict(next_states, verbose=0)
        best_next_actions = np.argmax(next_q_online, axis=1)

        next_q_target = self.target_model.predict(next_states, verbose=0)

        targets = self.model.predict(states, verbose=0)
        errors = np.zeros(self.batch_size)

        for i in range(self.batch_size):
            old_val = targets[i][actions[i]]
            if dones[i]:
                new_val = rewards[i]
            else:
                new_val = rewards[i] + self.gamma * next_q_target[i][best_next_actions[i]]
            targets[i][actions[i]] = new_val
            errors[i] = abs(old_val - new_val)

        # Update priorities
        self.memory.update_priorities(idxs, errors)

        # Train with importance-sampling weights
        self.model.fit(
            states, targets,
            sample_weight=is_weights,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
        )

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Anneal beta toward 1
        if self.beta < 1.0:
            self.beta = min(1.0, self.beta + self.beta_increment)

        # Periodic target-network sync
        if self.steps % self.target_update_freq == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        self.model.save_weights(path)

    def load(self, path: str):
        self.model.load_weights(path)
        self.update_target_model()
