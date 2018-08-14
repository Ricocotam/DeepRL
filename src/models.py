import random
import torch
import numpy as np
import util


def_device = torch.device("cpu")

class Model(object):
    """Base class for models."""
    def __init__(self, gamma, optim, loss_function, device=def_device):
        self.loss_function = loss_function
        self.optim = optim
        self.gamma = gamma
        self.device = device

    def __call__(self, state):
        pass

    def learn(self, sample):
        pass

    def update(self):
        pass


class DQN(Model):
    """docstring for DQN."""
    def __init__(self, net_structure, gamma, optim, loss_function, tau=1, device=def_device):
        super(DQN, self).__init__(gamma, optim, loss_function, device)

        self.predict = util.model_from_structure(net_structure).to(self.device)
        self.target = util.model_from_structure(net_structure).to(self.device)
        self.tau = tau
        self.optimizer = self.optim(self.predict.parameters())

        self.update_counter = 0

    def __call__(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.predict.eval()
        with torch.no_grad():
            action_values = self.predict(state)
        self.predict.train()
        return action_values

    def learn(self, sample):
        states, actions, rewards, next_states, dones = sample
        target_values = self.target(next_states).max(1)[0].detach().unsqueeze(1)
        expected_values = rewards + (self.gamma * target_values * (1-dones))

        actual_values = self.predict(states).gather(1, actions)

        loss = self.loss_function(actual_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.predict.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update(self):
        if self.update_counter > 0:
            for target_param, predict_param in zip(self.target.parameters(), self.predict.parameters()):
                target_param.data.copy_(self.tau*predict_param.data + (1.0-self.tau)*target_param.data)
        self.update_counter += 1
