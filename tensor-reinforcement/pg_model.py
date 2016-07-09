#-----------------------------
#Took Boilerplate code from here: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
#-----------------------------

import gym
import tensorflow as tf 
import numpy as np 
import random
from collections import deque
import pdb 
from train_stock import *

# Hyper Parameters for PG
GAMMA = 0.9 # discount factor for target Q 
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
BATCH_SIZE = 32 # size of minibatch
LEARNING_RATE = 1e-4

class PG():
    # DQN Agent
    def __init__(self, env):
        # init some parameters
        self.replay_buffer = []
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.n_input = self.state_dim
        n_hidden_1 = 60
        n_hidden_2 = 20
        
        weights = {
        'h1': tf.Variable(tf.random_normal([self.n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, self.action_dim]))
        }
        biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([self.action_dim]))
        }
        self.state_input = tf.placeholder("float", [None, self.n_input])
        self.y_input = tf.placeholder("float",[None, self.action_dim])
        self.create_pg_network(weights,biases)
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

        global summary_writer
        summary_writer = tf.train.SummaryWriter('logs',graph=self.session.graph)

    def create_pg_network(self, weights, biases):
        # network weights
        W1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        self.PG_value = tf.nn.softmax(tf.matmul(h_layer,W2) + b2)
        
    def create_training_method(self):
        #this needs to be updated to use softmax
        #P_action = tf.reduce_sum(self.PG_value,reduction_indices = 1)
        #self.cost = tf.reduce_mean(tf.square(self.y_input - P_action))
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.PG_value, self.y_input))
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_input * tf.log(self.PG_value), reduction_indices=[1]))
        tf.scalar_summary("loss",self.cost)
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def perceive(self,states,epd):
        temp = []
        for index, value in enumerate(states):
            temp.append([states[index], epd[index]])
        self.replay_buffer += temp

    def train_pg_network(self):
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE*5)
        state_batch = [data[0] for data in minibatch]
        y_batch = [data[1] for data in minibatch]
        #pdb.set_trace();
        self.optimizer.run(feed_dict={self.y_input:y_batch,self.state_input:state_batch})
        summary_str = self.session.run(merged_summary_op,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch
            })
        summary_writer.add_summary(summary_str,self.time_step)
        self.replay_buffer = []

        # save network every 1000 iteration
        if self.time_step % 1000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-pg', global_step = self.time_step)

    def policy_forward(self,state):
        prob = self.PG_value.eval(feed_dict = {self.state_input:[state]})[0]
        action = np.random.choice(self.action_dim, 1, p=prob)[0]
        y = np.zeros([self.action_dim])
        y[action] = 1
        return y, action

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def discounted_rewards(self,rewards):
        reward_discounted = np.zeros_like(rewards)
        track = 0
        for index in reversed(xrange(len(rewards))):
            track = track * GAMMA + rewards[index]
            reward_discounted[index] = track
        return reward_discounted


# ---------------------------------------------------------
ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = PG(env)
    state_list, reward_list, grad_list = [],[],[]
    episode_number = 0
    state = env.reset()

    while True:
        # initialize tase
        # Train 
        grad, action = agent.policy_forward(state) # e-greedy action for train
        state_list.append(state)
        state,reward,done,_ = env.step(action)
        #print(action)
        reward_list.append(reward)
        #print(reward)
        grad_list.append(grad)
        #print(grad_list)
        if done:
            episode_number += 1
            #print(episode_number)
            epr = np.vstack(reward_list)
            discounted_epr = agent.discounted_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            epdlogp = np.vstack(grad_list)
            epdlogp *= discounted_epr
            #print(epdlogp)
            agent.perceive(state_list, epdlogp)
            state = env.reset()
            state_list, reward_list, grad_list = [],[],[]
            if episode_number % BATCH_SIZE == 0:
              #train model
              agent.train_pg_network()
            #test every 100 rewards
            if episode_number % 100 == 0 and episode_number >= 100:
                total_reward = 0
                for i in xrange(TEST):
                    state = env.reset()
                    for j in xrange(STEP):
                        #env.render()
                        grad, action = agent.policy_forward(state) # direct action for test
                        state,reward,done,_ = env.step(action)
                        total_reward += reward
                        if done:
                            break
                ave_reward = total_reward/TEST
                print 'episode: ',episode_number,'Evaluation Average Reward:',ave_reward
                state = env.reset()

if __name__ == '__main__':
    main()