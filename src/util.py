import torch
import torch.nn as nn
import numpy as np
from collections import deque


def learn(env, goal_size, average_goal, agent, max_step, nb_epi_max, gamma, learning_policy):
    scores = []
    scores_window = deque(maxlen=goal_size)
    for i in range(nb_epi_max):
        state = env.reset()
        score = 0
        for _ in range(max_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = reward - 1
            agent.step((state, action, reward, next_state, done))
            score += reward
            state = next_state

            if done:
                break
        learning_policy.update()
        scores.append(score)
        scores_window.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
        if i % goal_size == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
        if np.mean(scores_window)>= average_goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-goal_size, np.mean(scores_window)))
            break
    return scores


def per_learn(env, goal_size, average_goal, agent, max_step, nb_epi_max, gamma, learning_policy):
    pass


def model_from_structure(net_structure, activation=nn.ReLU()):
    temp = []
    for prev, next in zip(net_structure[:-1], net_structure[1:]):
        temp.append(nn.Linear(prev, next))
        temp.append(activation)
    temp = temp[:-1]  # Remove last activation
    return nn.Sequential(*temp)
