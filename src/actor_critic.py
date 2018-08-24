import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
import random

from torch.distributions import Categorical

env = gym.make("LunarLander-v2")
eps = np.finfo(np.float32).eps.item()

# seed = 1155874
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# env.seed(seed)

class Policy(nn.Module):
    """docstring for Policy."""
    def __init__(self):
        super(Policy, self).__init__()
        self.net = nn.Sequential(nn.Linear(env.observation_space.shape[0], 128),
                                 nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(128, env.action_space.n),
                                    nn.Softmax(-1))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        temp = self.net(x)
        return self.policy(temp), self.critic(temp)


def play_episode(env, model, maxlen=10000, render=False):
    log_probs = []
    rewards = []
    values = []

    state = env.reset()
    for i in range(maxlen):
        if render:
            env.render()
        state = torch.from_numpy(state).float()
        action_prob, value = model(state)
        sampler = Categorical(action_prob)
        action = sampler.sample()
        state, reward, done, _ = env.step(action.item())
        log_probs.append(sampler.log_prob(action))
        values.append(value)
        rewards.append(reward)

        if done:
            break

    return log_probs, values, rewards

def learn(opti, log_probs, values, rewards):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + 0.99 * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + eps)
    policy_losses = []
    critic_losses = []
    for value, log_prob, reward in zip(values, log_probs, discounted_rewards):
        val = reward - value.item()
        current_loss = -log_prob * val
        policy_losses.append(current_loss)
        critic_losses.append(F.smooth_l1_loss(value, torch.tensor([reward])))

    policy_loss = torch.stack(policy_losses).sum()
    critic_loss = torch.stack(critic_losses).sum()
    loss = policy_loss + critic_loss
    opti.zero_grad()
    loss.backward()
    opti.step()


model = Policy()
opti = optim.Adam(model.parameters(), lr=3e-2)

gamma = 0.99
nb_epi = 9999999
goal_size = 100
average_goal = 200

scores_window = deque(maxlen=goal_size)
scores = []
for i in range(1, nb_epi):
    log_probs, values, rewards = play_episode(env, model)
    learn(opti, log_probs, values, rewards)
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

for i in range(10):
    play_episode(env, model, render=True)
env.close()
