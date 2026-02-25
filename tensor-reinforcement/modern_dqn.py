import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99    
        self.epsilon = 1.0   
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99 # Per episode decay now
        self.learning_rate = 0.00025
        self.batch_size = 128
        
        # Main Model (Policy Network)
        self.model = self._build_model()
        # Target Model (Stable Q-targets)
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        inputs = Input(shape=(self.state_dim,))
        
        # Shared feature extraction
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        
        # Value Stream
        value = Dense(64, activation='relu')(x)
        value = Dense(1, activation='linear')(value)
        
        # Advantage Stream
        advantage = Dense(64, activation='relu')(x)
        advantage = Dense(self.action_dim, activation='linear')(advantage)
        
        # Aggregation: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = Lambda(lambda args: args[0] + (args[1] - tf.reduce_mean(args[1], axis=1, keepdims=True)))([value, advantage])
        
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def _train_step(self, states, targets):
        self.model.optimizer.apply_gradients(zip(
            tf.gradients(self.model.loss(targets, self.model(states)), self.model.trainable_variables),
            self.model.trainable_variables
        ))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        # Using model() instead of model.predict() is much faster in TF 2.x
        act_values = self.model(state_tensor, training=False)
        return np.argmax(act_values[0])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batches
        states = np.array([i[0] for i in minibatch], dtype=np.float32)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch], dtype=np.float32)
        next_states = np.array([i[3] for i in minibatch], dtype=np.float32)
        dones = np.array([i[4] for i in minibatch], dtype=np.float32)

        # Faster inference using model call
        target_qs = self.target_model(next_states, training=False).numpy()
        targets = self.model(states, training=False).numpy()
        
        # Vectorized target update
        max_next_qs = np.max(target_qs, axis=1)
        target_updates = rewards + (1 - dones) * self.gamma * max_next_qs
        
        for i in range(self.batch_size):
            targets[i][actions[i]] = target_updates[i]
        
        # Single train_on_batch call is usually efficient enough
        self.model.train_on_batch(states, targets)

    def decay_epsilon(self):
        """Called outside training step to decay per episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
