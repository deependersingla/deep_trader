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
import sys
from tensorboard_helper import *

# Hyper Parameters for PG
GAMMA = 0.9 # discount factor for target Q 
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
BATCH_SIZE = 32 # size of minibatch
LEARNING_RATE = 1e-4

class PG():
    # DQN Agent
    def __init__(self, data_dictionary):
        # init some parameters
        self.replay_buffer = []
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = data_dictionary["input"]
        self.action_dim = data_dictionary["action"]
        self.n_input = self.state_dim
        self.state_input = tf.placeholder("float", [None, self.n_input])
        self.y_input = tf.placeholder("float",[None, self.action_dim])
        self.create_pg_network(data_dictionary)
        self.create_training_method()
        self.create_supervised_accuracy()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("pg_saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

        global summary_writer
        summary_writer = tf.train.SummaryWriter('logs',graph=self.session.graph)

    def create_pg_network(self, data_dictionary):
        # network weights
        W0 = self.weight_variable([self.state_dim,80])
        b0 = self.bias_variable([80])
        W1 = self.weight_variable([80,data_dictionary["hidden_layer_1_size"]])
        b1 = self.bias_variable([data_dictionary["hidden_layer_1_size"]])
        W2 = self.weight_variable([data_dictionary["hidden_layer_1_size"],data_dictionary["hidden_layer_2_size"]])
        b2 = self.bias_variable([data_dictionary["hidden_layer_2_size"]])
        W3 = self.weight_variable([data_dictionary["hidden_layer_2_size"],self.action_dim])
        b3 = self.bias_variable([self.action_dim])
        variable_summaries(b3, "layer2/bias")
        h_1_layer = tf.nn.relu(tf.matmul(self.state_input,W0) + b0)
        h_2_layer = tf.nn.relu(tf.matmul(h_1_layer,W1) + b1)
        h_layer = tf.nn.relu(tf.matmul(h_2_layer,W2) + b2)
        self.PG_value = tf.nn.softmax(tf.matmul(h_layer,W3) + b3)
        
    def create_training_method(self):
        #this needs to be updated to use softmax
        #P_action = tf.reduce_sum(self.PG_value,reduction_indices = 1)
        #self.cost = tf.reduce_mean(tf.square(self.y_input - P_action))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.PG_value, self.y_input))
        #self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_input * tf.log(self.PG_value), reduction_indices=[1]))
        tf.scalar_summary("loss",self.cost)
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def create_supervised_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.PG_value,1), tf.argmax(self.y_input,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
        if self.time_step % 100000 == 0:
            self.saver.save(self.session, 'pg_saved_networks/' + 'network' + '-pg', global_step = self.time_step)
    
    def train_supervised(self, state_batch, y_batch):
        #pdb.set_trace()
        self.optimizer.run(feed_dict={self.y_input:y_batch,self.state_input:state_batch})
        self.time_step += 1
        summary_str = self.session.run(merged_summary_op,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch
            })
        summary_writer.add_summary(summary_str,self.time_step)
        if self.time_step % 1000 == 0:
            self.saver.save(self.session, 'pg_saved_networks/' + 'network' + '-pg', global_step = self.time_step)

    def supervised_accuracy(self, state_batch, y_batch):
        return self.accuracy.eval(feed_dict={self.y_input:y_batch,self.state_input:state_batch})*100
    
    def policy_forward(self,state):
        prob = self.PG_value.eval(feed_dict = {self.state_input:[state]})[0]
        aprob = np.amax
        #print(action)
        if self.time_step > 20000000:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/9000000
        if random.random() <= 0:
            action = np.random.choice(self.action_dim, 1)[0]
        else:
            action = np.random.choice(self.action_dim, 1, p=prob)[0]       
        y = np.zeros([self.action_dim])
        self.time_step += 1
        y[action] = 1
        return y, action

    def action(self,state):
        prob = self.PG_value.eval(feed_dict = {self.state_input:[state]})[0]
        action = np.argmax(prob)
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
EPISODE = 10000 # Episode limitation
STEP = 9 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
ITERATION = 20

def main():
    # initialize OpenAI Gym env and dqn agent
    episode_number = 0
    data_dictionary = get_intial_data()
    agent = PG(data_dictionary)
    test_rewards = {}

    #supervised learning first
    supervised_seeding(agent, data_dictionary)

    for iter in xrange(ITERATION):
        print(iter)
        # initialize tase
        # Train 
        data = data_dictionary["x_train"]
        for episode in xrange(len(data)):
            episode_data = data[episode]
            state_list, reward_list, grad_list = [],[],[]
            portfolio = 0
            portfolio_value = 0
            for step in xrange(STEP):
                state, action, next_state, reward, done, portfolio, portfolio_value, grad = env_stage_data(agent, step, episode_data, portfolio, portfolio_value, True)
                state_list.append(state)
                grad_list.append(grad)
                reward_list.append(reward)
                if done:
                    epr = np.vstack(reward_list)
                    discounted_epr = agent.discounted_rewards(epr)
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)
                    epdlogp = np.vstack(grad_list)
                    agent.perceive(state_list, epdlogp)
                    if episode % BATCH_SIZE == 0 and episode > 1:
                        agent.train_pg_network()
                    break
            if episode % 100  == 0 and episode > 1:
                total_reward = 0
                for i in xrange(10):
                    for step in xrange(STEP):
                        state, action, next_state, reward, done, portfolio, portfolio_value, grad = env_stage_data(agent, step, episode_data, portfolio, portfolio_value, True)
                        #pdb.set_trace();
                        total_reward += reward
                        if done:
                            break
                ave_reward = total_reward/10
                #print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
        #on test data
        data = data_dictionary["x_test"]
        iteration_reward = []
        for episode in xrange(len(data)):
            episode_data = data[episode]
            portfolio = 0
            portfolio_list = []
            portfolio_value = 0
            portfolio_value_list = []
            reward_list = []
            total_reward = 0
            action_list = []
            for step in xrange(STEP):
                state, action, next_state, reward, done, portfolio, portfolio_value, grad = env_stage_data(agent, step, episode_data, portfolio, portfolio_value, False)
                action_list.append(action)
                portfolio_list.append(portfolio)
                portfolio_value_list.append(portfolio_value)
                reward_list.append(reward)
                total_reward += reward
                if done:
                    episode_reward = show_trader_path(action_list, episode_data, portfolio_list, portfolio_value_list, reward_list)
                    iteration_reward.append(episode_reward)
                    break
            #print 'episode: ',episode,'Testing Average Reward:',total_reward
        avg_reward = sum(iteration_reward) # / float(len(iteration_reward))
        print(avg_reward)
        test_rewards[iter] = [iteration_reward, avg_reward]
    for key, value in test_rewards.iteritems():
        print(value[0])
    for key, value in test_rewards.iteritems():
        print(key)
        print(value[1])


def supervised_seeding(agent, data_dictionary):
    for iter in xrange(ITERATION):
        print("Iteration:")
        print(iter)
        iteration_accuracy = []
        train_iteration_accuracy = []
        data = data_dictionary["x_train"]
        y_label_data = data_dictionary["y_train"]
        for episode in xrange(len(data)):
            state_batch, y_batch = make_supervised_input_vector(episode, data, y_label_data)
            #print(episode)
            agent.train_supervised(state_batch, y_batch)
            accuracy = agent.supervised_accuracy(state_batch, y_batch)
            train_iteration_accuracy.append(accuracy)
        avg_accuracy = sum(train_iteration_accuracy)/ float(len(train_iteration_accuracy))
        print("Train Average accuracy")
        print(avg_accuracy)

        data = data_dictionary["x_test"]
        y_label_data = data_dictionary["y_test"]
        for episode in xrange(len(data)):
            #pdb.set_trace();
            state_batch, y_batch = make_supervised_input_vector(episode, data, y_label_data)
            accuracy = agent.supervised_accuracy(state_batch, y_batch)
            iteration_accuracy.append(accuracy)
        avg_accuracy = sum(iteration_accuracy) / float(len(iteration_accuracy))
        print("Test Average accuracy")
        print(avg_accuracy)




def make_supervised_input_vector(episode, data, y_label_data):
    x_data = data[episode]
    y_data = y_label_data[episode]
    state_batch = []
    y_batch = []
    for index, item in enumerate(x_data[0:-1]):
        try:
            temp = item + [y_data[index][1]]
        except:
            pdb.set_trace()
        state_batch.append(temp)
        y = np.zeros([3])
        y[y_data[index][0]] = 1
        y_batch.append(y)
    return state_batch, y_batch








def env_stage_data(agent, step, episode_data, portfolio, portfolio_value, train):
    state = episode_data[step] + [portfolio]
    if train:
        grad, action = agent.policy_forward(state) # e-greedy action for train
    else:
        grad, action = agent.action(state)
    #print(step)
    new_state = episode_data[step+1]
    if step == STEP - 1:
        done = True
    else:
        done = False
    next_state,reward,done,portfolio,portfolio_value = new_stage_data(action, portfolio, state, new_state, portfolio_value, done, episode_data[step])
    return state, action, next_state, reward, done, portfolio, portfolio_value, grad

if __name__ == '__main__':
    main()