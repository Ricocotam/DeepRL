import random
import torch
import numpy as np

from collections import deque, namedtuple

import policies
import buffer as bf

def_device = torch.device("cpu")

class Agent(object):
    """A base class for deep RL agents."""
    def __init__(self, model, policy_learning, policy_playing=policies.Greedy(), buffer=bf.SoloBuffer(), learn_every=1, update_every=1):
        """Initialize an agent.

        Parameters
        -------------
            model : Model
                The model you want to use

            buffer : ReplayBuffer
                The buffer you want to use for your episodes

            lean_every : int
                The step gap between two learning phase

            update_every : int
                The step gap between two updating phase

            policy_learning : function(state, model)
                The policy you use during learning

            policy_playing : function(state, model)
                The policy used during playing phase
        """
        self.model = model
        self.buffer = buffer
        self.learning_strategy = policy_learning
        self.playing_strategy = policy_playing
        self.update_every = update_every
        self.learn_every = learn_every
        self.update_counter = 0
        self.learn_counter = 0
        self.learning = True

    def act(self, state):
        """Get the action to play."""
        if self.learning:
            return self.learning_strategy(state, self.model)
        else:
            return self.playing_strategy(state, self.model)

    def step(self, experience):
        """Do a step for the agent. Memorize and learn if needed."""
        self.buffer.add(experience)
        self.update_counter = (self.update_counter + 1) % self.update_every
        self.learn_counter = (self.learn_counter + 1) % self.learn_every

        if self.buffer.can_sample():
            if self.learn_counter == 0:
                sample = self.buffer.sample()
                self.model.learn(sample)

            if self.update_counter == 0:
                self.model.update()

    def learning(self):
        """Set learning policy."""
        self.learning = True

    def playing(self):
        """Set playing policy."""
        self.learning = False


class PERAgent(Agent):
    """docstring for PERAgent."""
    def __init__(self, model, policy_learning, policy_playing=policies.Greedy(), buffer=bf.SoloBuffer, learn_every=1, update_every=1):
        super(PERAgent, self).__init__(model, policy_learning, policy_playing, buffer, learn_every, update_every)

    def step(self, experience):
        """Do a step for the agent. Memorize and learn if needed."""
        self.buffer.add(experience)
        self.update_counter = (self.update_counter + 1) % self.update_every
        self.learn_counter = (self.learn_counter + 1) % self.learn_every
        if self.buffer.can_sample():
            if self.learn_counter == 0:
                sample, indices, weights = self.buffer.sample()
                diff = self.model.learn([sample, weights])
                self.buffer.update(indices, diff)


            if self.update_counter == 0:
                self.model.update()
