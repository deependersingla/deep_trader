## this file take input from ib and then make vector and run the experiment
##it takes care of filtering epsilon etx

from dqn_training import *
import pdb


class Agent: 
    policyFrozen = False

    def __init__(self, lastAction, input_vector_length):
        # lastAction should be [0]
        self.lastAction = lastAction

        self.time = 0
        self.epsilon = 1.0  # Initial exploratoin rate

        # Pick a DQN from DQN_class
        self.input_vector_length = input_vector_length
        self.DQN = DQN_class(input_vector_length = input_vector_length)  # Default is for "Pong".

    def agent_start(self, observation):

        # Get data from current observation array
        # Initialize State
        #here observation is 80 neurons, 5 * 14 (with 10 temporality) + 5 (of last one hour) + 5 (of last 24 hour)
        self.state = observation
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1, self.input_vector_length), dtype=np.float32))

        # Generate an Action e-greedy
        action, Q_now = self.DQN.e_greedy(state_, self.epsilon)

        # Update for next step
        self.lastAction = copy.deepcopy(action)
        self.last_state = self.state.copy()
        self.last_observation = observation

        return action

    def agent_step(self, reward, observation):

        # Preproces
        self.state = observation
        state_ = cuda.to_gpu(np.asanyarray(self.state.reshape(1, self.input_vector_length), dtype=np.float32))

        # Exploration decays along the time sequence
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.DQN.initial_exploration < self.time:
                self.epsilon -= 1.0/10**6
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print "Initial Exploration : %d/%d steps" % (self.time, self.DQN.initial_exploration)
                eps = 1.0
        else:  # Evaluation
                print "Policy is Frozen"
                eps = 0.05

       
        action, Q_now = self.DQN.e_greedy(state_, eps)

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DQN.stockExperience(self.time, self.last_state, self.lastAction, reward, self.state, False)
            self.DQN.experienceReplay(self.time)

        if self.DQN.initial_exploration < self.time and np.mod(self.time, self.DQN.target_model_update_freq) == 0:
            print "########### MODEL UPDATED ######################"
            self.DQN.target_model_update()

        # Simple text based visualization
        print ' Time Step %d /   ACTION  %d  /   REWARD %.1f   / EPSILON  %.6f  /   Q_max  %3f' % (self.time, self.DQN.action_to_index(action), np.sign(reward), eps, np.max(Q_now.get()))

        # Updates for next step
        self.last_observation = observation
        
        # Update for next step
        if self.policyFrozen is False:
            self.lastAction = copy.deepcopy(action)
            self.last_state = self.state.copy()
            self.time += 1

        return action

    def agent_end(self, reward):  # Episode Terminated

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.DQN.stockExperience(self.time, self.last_state, self.lastAction, reward, self.last_state, True)
            self.DQN.experienceReplay(self.time)

        # Simple text based visualization
        print '  REWARD %.1f   / EPSILON  %.5f' % (np.sign(reward), self.epsilon)

        # Time count
        if not self.policyFrozen:
            self.time += 1

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        if inMessage.startswith("freeze learning"):
            self.policyFrozen = True
            return "message understood, policy frozen"

        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen = False
            return "message understood, policy unfrozen"

        if inMessage.startswith("save model"):
            with open('dqn_model.dat', 'w') as f:
                pickle.dump(self.DQN.model, f)
            return "message understood, model saved"