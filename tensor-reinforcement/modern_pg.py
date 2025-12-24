import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import random

class PGAgent:
    """
    Policy Gradient (REINFORCE) Agent using TensorFlow 2.
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        
        self.model = self._build_model()
        self.optimizer = Adam(learning_rate=self.lr)
        
        # Memory for the current episode
        self.states = []
        self.actions = []
        self.rewards = []

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        # Softmax for action probabilities
        model.add(Dense(self.action_dim, activation='softmax'))
        return model

    def act(self, state):
        # Forward pass to get probabilities
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.model(state_tensor)
        probs = probs.numpy()[0]
        
        # Sample action from distribution
        action = np.random.choice(self.action_dim, p=probs)
        return action

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def train(self):
        """
        Updates the policy network using the REINFORCE algorithm 
        at the end of the episode.
        """
        # Calculate discounted rewards (returns)
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add
            
        # Normalize returns for stability
        if len(discounted_rewards) > 1:
            mean = np.mean(discounted_rewards)
            std = np.std(discounted_rewards) + 1e-8
            discounted_rewards = (discounted_rewards - mean) / std

        # Convert to tensors
        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        returns = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

        # Gradient Descent
        with tf.GradientTape() as tape:
            # Get probabilities for all states in batch
            all_probs = self.model(states)
            
            # Select probabilities of the actions actually taken
            # We use one_hot encoding to mask
            action_masks = tf.one_hot(actions, self.action_dim)
            prob_actions_taken = tf.reduce_sum(all_probs * action_masks, axis=1)
            
            # Log probabilities
            log_probs = tf.math.log(prob_actions_taken + 1e-8)
            
            # Loss = -mean(log_prob * return)
            loss = -tf.reduce_mean(log_probs * returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        return loss.numpy()
