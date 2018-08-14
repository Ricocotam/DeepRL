import random
import torch
import numpy as np

from collections import deque, namedtuple

import policies

def_device = torch.device("cpu")


class Agent(object):
    """A base class for deep RL agents."""
    def __init__(self, model, policy_learning, policy_playing=policies.Greedy(), buffer=SoloBuffer(), learn_every=1, update_every=1):
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

class ReplayBuffer(object):
    """docstring for ReplayBuffer."""
    def __init__(self, buffer_size, batch_size, device=def_device):
        super(ReplayBuffer, self).__init__()
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, step):
        pass

    def sample(self):
        pass

    def can_sample(self):
        return len(self.memory) > self.batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SoloBuffer(ReplayBuffer):
    """docstring for SoloBuffer."""
    def __init__(self, device=def_device):
        super(SoloBuffer, self).__init__(0, 0, device)
        self.memory = None
        self.batch_size = batch_size
        self.device = device

    def add(self, step):
        self.memory = step

    def sample(self):
        return self.memory

    def can_sample(self):
        return self.memory is not None

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device=def_device):
        """Initialize a ReplayBuffer object.

        Params
        --------
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        super(QBuffer, self).__init__(buffer_size, batch_size, device)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, step):
        """Add a new experience to memory."""
        self.memory.append(self.experience(*step))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

class CompleteBuffer(ReplayBuffer):
    """docstring for CompleteBuffer."""
    def __init__(self, buffer_size, batch_size, device=def_device):
        super(CompleteBuffer, self).__init__(buffer_size, batch_size, device)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "next_action", "done"])

    def add(self, step):
        """Add a new experience to memory."""
        self.memory.append(self.experience(*step))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        next_actions = torch.from_numpy(np.vstack([e.next_action for e in experiences if e is not None])).long().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, next_actions, dones)
