import torch

from constants import *

# gets normalized generalized advantage estimates
def get_advs(rewards, values):
    advantages = torch.zeros_like(rewards)
    for t in range(len(rewards)):
        adv = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t + l] + \
                    GAMMA * values[t + l + 1] - values[t + 1]
            adv += ((GAMMA * TAU) ** l) * delta
        adv += ((GAMMA * TAU) ** l) * (rewards[t + l] - values[t + l])
        advantages[t] = adv
    return normalize(advantages)

# Discount future rewards (reward to go)
def get_RTG(rewards):
    total = 0
    rtg = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        total = total * GAMMA + rewards[t]
        rtg[t] = total
    return rtg

# Normalize tensor by its mean and std
def normalize(t):
    return (t - t.mean()) / (t.std() + 1e-8)

def makeMask(done):
    mask = torch.tensor([1.0 - done], device = 'cuda')
    return mask

def generate_indices(total_size, batch_size):
    inds = torch.randperm(total_size)
    return [inds[i * batch_size:(i+1) * batch_size] for i
            in range(0, total_size // batch_size)]

def reduce_tensor(t):
    if type(t) is torch.Tensor:
        return t.squeeze()
    else:
        return t

class RolloutStorage:
    def  __init__(self, MAX_SIZE = 4096):
        # Using empty tensors makes it easier to assert
        # shape on insertions
        self.store = None
        self.size = None
        self.MAX_SIZE = MAX_SIZE
        self.reset()

    def remember(self, log_prob, value, state, action, reward):
        if self.size == self.MAX_SIZE:
            return
        size = self.size

        self.store['log_prob'][size] = reduce_tensor(log_prob)
        self.store['value'][size] = reduce_tensor(value)
        self.store['state'][size] = reduce_tensor(state)
        self.store['action'][size] = reduce_tensor(action)
        self.store['reward'][size] = reduce_tensor(reward)
        self.size += 1

    def detach(self):
        for key in self.store.keys():
            self.store[key] = self.store[key].detach()

    def cuda(self):
        for key in self.store.keys():
            self.store[key] = self.store[key].to('cuda')

    def reset(self):
        MAX_SIZE = self.MAX_SIZE
        self.store = {'log_prob' : torch.zeros(MAX_SIZE),
                'value' : torch.zeros(MAX_SIZE),
                'state' : torch.zeros(MAX_SIZE, STATE_DIM), 
                'action' : torch.zeros(MAX_SIZE),
                'reward' : torch.zeros(MAX_SIZE)}
        self.size = 0

    def set_terminal(self, last_v):
        self.store['value'][self.size] = last_v

    def get(self, key):
        if key == 'value':
            return self.store['value'][0:self.size+1]
        else:
            return self.store[key][0:self.size]
