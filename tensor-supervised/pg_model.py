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
    def __init__(self):
        # init some parameters
        self.replay_buffer = deque()
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
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        self.PG_value = tf.matmul(layer_2, weights['out']) + biases['out']
        
    def create_training_method(self):
        self.state_input = tf.placeholder("float", [None, self.n_input])
        self.y_input = tf.placeholder("float", [None, self.action_dim])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.PG_value, self.y_input))
        tf.scalar_summary("loss",self.cost)
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

    def perceive(self,states,epd):
        self.replay_buffer.append((states,epd))

    def train_pg_network(self):
        
        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch
            })
        summary_str = self.session.run(merged_summary_op,feed_dict=feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch
            })
        summary_writer.add_summary(summary_str,self.time_step)

        # save network every 1000 iteration
        if self.time_step % 1000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-pg', global_step = self.time_step)

    def policy_forward(self,state):
        PG_value = self.PG_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        prob = np.amax(PG_value)
        action = np.argmax(PG_value)
        if np.random.uniform() < prob:
            grad = 1 - prob
        else:
            grad = -prob
            temp_list = xrange(len(PG_value))
            del temp_list [action]
            action = np.random.choice(temp_list)
        return prob, action, grad

    def action(self,state):
        return np.argmax(self.PG_value.eval(feed_dict = {
            self.state_input:[state]
            })[0])

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def discounted_rewards(rewards):
        reward_discounted = np.zeros_like(rewards)
        track = 0
        for index in reversed(xrange(len(rewards))):
            track = track * GAMMA + rewards[index]
            reward_discounted[index] = track
        return reward_discounted


# ---------------------------------------------------------
ENV_NAME = 'Pong-v0'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = PG(env)
    state_list, reward_list, grad_list = [],[],[]
    episode_number = 0

    while True:
        # initialize task
        state = env.reset()
        # Train 
        prob, action, grad = agent.policy_forward_greedy(state) # e-greedy action for train
        state_list.append(state)
        state,reward,done,_ = env.step(action)
        reward_list.append(reward)
        grad_list.append(grad)
        if done:
            episode_number += 1
            epx = np.vstack(state_list)
            epr = np.vstack(reward_list)
            epd = np.vstack(grad_list)
            discounted_epr = discounted_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            epd *= discounted_epr
            agent.perceive(epx,epd)
            break
        if episode_number % BATCH_SIZE == 0:
            #train model
            train_pg_network(self)
            self.replay_buffer = deque()
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(STEP):
                    env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
            if ave_reward >= 200:
                break

if __name__ == '__main__':
    main()