import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
import random
from per_buffer import PrioritizedReplayBuffer

class RainbowAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025 # Lower LR often better for Double/Dueling
        self.batch_size = 64
        self.update_target_freq = 1000 # Steps
        
        # PER Hyperparameters
        self.beta = 0.4 # Initial importance sampling weight
        self.beta_increment = 0.001
        
        # Memory
        self.memory = PrioritizedReplayBuffer(capacity=20000)
        
        # Models
        self.model = self._build_dueling_model()
        self.target_model = self._build_dueling_model()
        self.update_target_model()
        
        self.steps = 0

    def _build_dueling_model(self):
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
        # We use Lambda layer to implement the aggregation logic custom
        def aggregation(args):
            v, a = args
            return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
            
        q_values = Lambda(aggregation)([value, advantage])
        
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        self.steps += 1
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32)
        act_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(act_values[0])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from PER
        minibatch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
        
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Double DQN Logic
        # 1. Select best action using Online Model
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        
        # 2. Evaluate that action using Target Model
        target_next_qs = self.target_model.predict(next_states, verbose=0)
        
        # 3. Construct Targets
        targets = self.model.predict(states, verbose=0)
        # We need absolute errors for PER update
        errors = np.zeros(self.batch_size)
        
        for i in range(self.batch_size):
            old_val = targets[i][actions[i]]
            if dones[i]:
                new_val = rewards[i]
            else:
                # Double Q: Q_target(s', argmax Q_online(s', a'))
                new_val = rewards[i] + self.gamma * target_next_qs[i][next_actions[i]]
            
            targets[i][actions[i]] = new_val
            errors[i] = np.abs(old_val - new_val) # TD Error
            
        # Update Priorities in Buffer
        self.memory.update_priorities(idxs, errors)
        
        # Train with Importance Sampling Weights
        # Keras fit/train_on_batch supports sample_weight
        self.model.fit(states, targets, sample_weight=is_weights, batch_size=self.batch_size, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Anneal beta
        if self.beta < 1.0:
            self.beta += self.beta_increment
