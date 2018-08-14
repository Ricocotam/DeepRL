import torch
import torch.nn as nn
import torch.optim as optim

import gym

from torch.distributions import Categorical

env = gym.make("LunarLander-v2")

class Policy(nn.Module):
    """docstring for Policy."""
    def __init__(self):
        super(Policy, self).__init__()
        self.net = nn.Sequential(nn.Linear(env.observation_space.shape[0], 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, env.action_space.n),
                                 nn.Softmax()).double()

    def forward(self, x):
        return self.net(x)


model = Policy()
opti = optim.Adam(model.parameters())

gamma = 0.99
nb_epi = 501

for i in range(nb_epi):
    print(i)
    # Play
    log_probs = []
    rewards = []
    state = env.reset()
    for _ in range(501): # No inf loop
        if (i % 50) == 0:
            env.render()

        action_prob = model(torch.tensor(state))
        sampler = Categorical(action_prob)
        action = sampler.sample()
        log_probs.append(sampler.log_prob(action))
        next_state, reward, done, _ = env.step(action.item())
        rewards.append(reward)

        state = next_state
        if done:
            break

    # Compute discounted reward
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.append(R)
    discounted_rewards = torch.tensor(discounted_rewards[::-1]).double()
    print("Reward :", discounted_rewards[0])
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
