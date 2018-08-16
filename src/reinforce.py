import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import deque
from torch.distributions import Categorical

from models import Model

env = gym.make("LunarLander-v2")

class Policy(nn.Module):
    """docstring for Policy."""
    def __init__(self):
        super(Policy, self).__init__()
        self.net = nn.Sequential(nn.Linear(env.observation_space.shape[0], 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, env.action_space.n),
                                 nn.Softmax()).double()

    def forward(self, x):
        return self.net(x)

def play_episode(env, model, maxlen=99999999999):
    log_probs = []
    rewards = []
    state = env.reset()
    for i in range(maxlen):
        action_prob = model(torch.tensor(state))
        sampler = Categorical(action_prob)
        action = sampler.sample()
        next_state, reward, done, _ = env.step(action.item())

        log_probs.append(sampler.log_prob(action))
        rewards.append(reward)

        state = next_state
        if done:
            break
    return log_probs, rewards


def learn(opti, log_probs, rewards):
    # Compute discounted reward
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.append(R)
    discounted_rewards = torch.tensor(discounted_rewards[::-1]).double()
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-3)

    losses = []
    for t in range(len(rewards)):
        reward = discounted_rewards[t]
        log_prob = log_probs[t]
        current_loss = -log_prob * reward
        losses.append(current_loss.unsqueeze(0))

    policy_loss = torch.cat(losses).sum()
    opti.zero_grad()
    policy_loss.backward()
    opti.step()


model = Policy()
opti = optim.Adam(model.parameters())

gamma = 0.99
nb_epi = 999999999

goal_size = 100
average_goal = 200

scores_window = deque(maxlen=goal_size)
scores = []
for i in range(nb_epi):
    log_probs, rewards = play_episode(env, model)
    learn(opti, log_probs, rewards)
    scores.append(sum(rewards))
    scores_window.append(sum(rewards))

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
    if i % goal_size == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
    if np.mean(scores_window)>= average_goal:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-goal_size, np.mean(scores_window)))
        break
plt.plot(scores)
plt.show()
