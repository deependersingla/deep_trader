import numpy as np
import pdb

def generate_actions_from_price_data(prices):
    old_profit = 0
    golden_actions = []
    for action_list in global_array:
        #the below method can also be replaces with algorithms which generates action non-iteravely but 
        #i am too lazy to do that and got lot of computation power
        profit, result_list = find_profit_from_given_action(prices, action_list)
        if profit >= old_profit:
            old_profit = profit
            golden_actions = result_list
    print(golden_actions)
    return golden_actions


def iteration_based_result():
    profit = 0
    #for max profit, machine don't have to hold as it knows future price it just have to sell and buy
    total_iteration_list = []
    action_list = [1,2]
    episode = 9
    global_array = []
    #pdb.set_trace();
    get_iteration_actions_recursive(action_list, [], episode, global_array)
    return global_array

def find_profit_from_given_action(prices, actions):
    portfilio = 0
    portfilio_value = 0
    result_list = []
    for index, action in enumerate(actions):
        price = prices[index]
        if action == 1: #buy
            portfilio += 1
            portfilio_value -= price
        elif action == 2: #sell
            portfilio -= 1
            portfilio_value += price
        result_list.append([action, portfilio])
    profit = portfilio_value + (portfilio) * prices[-1]
    return profit, result_list


def get_iteration_actions_recursive(action_list, temp_array, episode, global_array):
    #base case
    if episode == 0:
        global_array.append(temp_array)
        return global_array
    for i in action_list:
        new_temp_array = temp_array + [i]
        get_iteration_actions_recursive(action_list, new_temp_array, episode -1, global_array)
global_array = iteration_based_result()