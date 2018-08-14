import random
import torch
import numpy as np
import torch.nn.functional as F


class Policy(object):
    """Base class for policies."""
    def __init__(self):
        pass

    def __call__(self, state, model):
        pass

    def update(self): # Call after each episode
        pass


class Greedy(Policy):
    """Classic greedy policy."""
    def __init__(self):
        pass

    def __call__(self, state, model):
        return model(state).argmax().tolist()


class EpsGreedy(Policy):
    """Classical epsilon-greedy policy."""
    def __init__(self, eps, action_size):
        self.eps = eps
        self.action_size = action_size

    def __call__(self, state, model):
        if random.random() < self.eps:
            return random.randint(0, self.action_size-1)
        else:
            return model(state).argmax().tolist()


class EpsDecay(EpsGreedy):
    """Epsilon greedy with a decay."""
    def __init__(self, eps_start, eps_min, eps_decay, action_size):
        super(EpsDecay, self).__init__(eps_start, action_size)
        self.eps_min = eps_min
        self.decay = eps_decay

    def update(self):
        self.eps = max(self.eps_min, self.eps * self.decay)

class SoftmaxPolicy(Policy):
    """docstring for SoftmaxPolicy."""
    def __init__(self):
        pass

    def __call__(self, state, model):
        probs = F.softmax(model(state), dim=1).numpy()[0]
        return random.choices(range(len(probs)), weights=probs, k=1)[0]
