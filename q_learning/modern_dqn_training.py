import numpy as np
import sys
import os

# Import the modern implementation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tensor-reinforcement')))
from modern_dqn import DQNAgent

class DQN_class:
    """
    Wrapper around the modern Keras DQNAgent to maintain compatibility 
    with the legacy q_learning interface (originally Chainer-based).
    """
    initial_exploration = 1000
    target_model_update_freq = 100

    def __init__(self, input_vector_length, enable_controller=[0, 1, 2]):
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller
        self.input_vector_length = input_vector_length
        
        # Instantiate modern agent
        self.agent = DQNAgent(state_dim=input_vector_length, action_dim=self.num_of_actions)
        print("Initialized Modern Keras DQN Backend for Legacy Wrapper.")

    def e_greedy(self, state, epsilon):
        """
        Original expected input: state (on GPU/Chainer variable)
        New input: state (numpy array)
        """
        # Ensure state is numpy (it might come as a list or array)
        if hasattr(state, 'get'): # wrapper for GPU array
             state = state.get()
        
        state = np.array(state).flatten()
        
        # Override agent's epsilon for this call
        self.agent.epsilon = epsilon
        
        # Get action
        action_idx = self.agent.act(state)
        
        # Get Q-values for debug/display (Original returned Q_now)
        # We need to run predict manually
        state_tensor = np.expand_dims(state, axis=0)
        q_values = self.agent.model.predict(state_tensor, verbose=0)[0]
        
        # Return Dummy Wrapper to simulate Chainer variable '.get()' API if needed, 
        # or just return numpy array if the caller handles it.
        # Original code: np.max(Q_now.get())
        class MockVariable:
            def get(self): return q_values
            
        return self.index_to_action(action_idx), MockVariable()

    def stockExperience(self, time, state, action, reward, state_dash, episode_end_flag):
        """
        Store experience.
        Original Chainer code stored in self.D.
        Modern agent has self.memory.
        """
        # Convert action to index
        try:
            action_idx = self.action_to_index(action)
        except:
            action_idx = action # fallback
            
        self.agent.remember(state, action_idx, reward, state_dash, episode_end_flag)

    def experienceReplay(self, time):
        """
        Trigger training.
        """
        self.agent.replay()

    def target_model_update(self):
        self.agent.update_target_model()

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)
