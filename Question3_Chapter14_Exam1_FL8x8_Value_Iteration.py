# -*- coding: utf-8 -*-
"""Chapter14_Exam1_FL_value_iteration8x8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aUz2rWTmen_bZJo8SZ0doZpY3dEjQ6e_

'''Chapter14_EXAM1_FL_Value_Iteration_8x8 
Chapter14 Homework 
DeepLearning By Dr.Rahmani
Wroten By: Hossein.Moharrer.Derakhshandeh
'''
"""

import torch
import gym

env = gym.make('FrozenLake8x8-v0')
env.reset()
env.render()
gamma = 0.99
threshold = 0.0001

def value_iteration(env, gamma, threshold):
    """
    Solve a given environment with value iteration algorithm
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the optimal policy for the given environment
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.empty(n_state)
        for state in range(n_state):
            v_actions = torch.zeros(n_action)
            for action in range(n_action):
                for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                    v_actions[action] += trans_prob * (reward + gamma * V[new_state])
            V_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V

V_optimal = value_iteration(env, gamma, threshold)
print('Optimal values:\n', V_optimal)

def extract_optimal_policy(env, V_optimal, gamma):
    """
    Obtain the optimal policy based on the optimal values
    @param env: OpenAI Gym environment
    @param V_optimal: optimal values
    @param gamma: discount factor
    @return: optimal policy
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    optimal_policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * V_optimal[new_state])
        optimal_policy[state] = torch.argmax(v_actions)
    return optimal_policy

optimal_policy = extract_optimal_policy(env, V_optimal, gamma)
print('Optimal policy:\n', optimal_policy)

def run_episode(env, policy):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward


n_episode = 1000
total_rewards = []
for episode in range(n_episode):
    total_reward = run_episode(env, optimal_policy)
    total_rewards.append(total_reward)

print('Average total reward under the optimal policy:', sum(total_rewards) / n_episode)