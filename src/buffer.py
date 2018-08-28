import random
import torch
import numpy as np

from collections import deque, namedtuple


def_device = torch.device("cpu")

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
        self.batch_size = 1
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

class PrioritizedBuffer(ReplayBuffer):
    """docstring for PrioritizedBuffer."""
    def __init__(self, buffer_size, batch_size, device=def_device, alpha=0.6, beta_start=0.4, beta_fac=(1e-3 / 200)):
        super(PrioritizedBuffer, self).__init__(buffer_size, batch_size, device)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.prob_alpha = alpha
        self.priorities = deque(maxlen=buffer_size)
        self.beta = beta_start
        self.beta_fac = beta_fac

    def add(self, step):
        self.memory.append(self.experience(*step))
        try:
            max_prio = max(self.priorities)
        except ValueError as e:
            max_prio = 1.0**self.prob_alpha
        self.priorities.append(max_prio)

    def sample(self):
        batch_size = self.batch_size
        prios = np.asarray(self.priorities)
        total = len(self.memory)
        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        self.beta = beta = min(1.0, self.beta*self.beta_fac)

        #min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min*total)**(-beta)

        weights  = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights  = torch.tensor(weights, device=self.device, dtype=torch.float)

        return (states, actions, rewards, next_states, dones), indices, weights

    def update(self, batch_indices, batch_delta):
        for idx, prio in zip(batch_indices, batch_delta):
            self.priorities[idx] = (prio.numpy() + 1e-5)**self.prob_alpha
