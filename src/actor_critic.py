import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import gym

from torch.distributions import Categorical

env = gym.make("LunarLander-v2")
env.seed(543)
torch.manual_seed(543)
np.random.seed(543)
random.seed(543)

class Policy(nn.Module):
    """docstring for Policy."""
    def __init__(self):
        super(Policy, self).__init__()
        self.net = nn.Sequential(nn.Linear(env.observation_space.shape[0], 128),
                                 nn.ReLU()).double()
        self.policy = nn.Sequential(nn.Linear(128, env.action_space.n),
                                    nn.Softmax(-1)).double()
        self.critic = nn.Linear(128, 1).double()

    def forward(self, x):
        temp = self.net(x)
        return self.policy(temp), self.critic(temp)


model = Policy()
opti = optim.Adam(model.parameters(), lr=0.03)

gamma = 0.99
nb_epi = 501

for i in range(nb_epi):
    print(i, end=" ")
    # Play
    log_probs = []
    rewards = []
    state_values = []
    state = env.reset()
    for _ in range(10000): # No inf loop
        if (i % 50) == 0:
            env.render()

        action_prob, critic = model(torch.tensor(state).double())
        sampler = Categorical(action_prob)
        action = sampler.sample()

        state, reward, done, _ = env.step(action.item())

        log_probs.append(sampler.log_prob(action))
        state_values.append(critic)
        rewards.append(reward)

        if done:
            break

    # Compute discounted reward
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards).double()
    print("Reward :", sum(rewards))
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

    policy_losses = []
    critic_losses = []
    for value, log_prob, reward in zip(state_values, log_probs, discounted_rewards):
        val = r - value.item()
        current_loss = -log_prob * val
        policy_losses.append(current_loss)
        critic_losses.append(F.smooth_l1_loss(value, reward))

    policy_loss = torch.stack(policy_losses).sum()
    critic_loss = torch.stack(critic_losses).sum()
    loss = policy_loss + critic_loss
    opti.zero_grad()
    loss.backward()
    opti.step()
