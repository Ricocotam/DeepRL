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
