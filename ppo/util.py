import torch

from constants import *

# get generalized advantage estimate with next value, rewards, masks and values
def get_GAE(next_v, r, m, v):
    v = v + [next_v]
    gae = 0
    returns = []
    # Iterate backwards through rollout
    for t in reversed(range(len(r))):
        delta = r[t] + GAMMA * v[t + 1] * m[t] - v[t]
        gae = delta + GAMMA * TAU * m[t] * gae
        # Since iteration is reversed, add in reversed order to undo
        returns.insert(0, gae + v[t])

    return returns

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

NUM_ITEMS = 6 # Different items to store in rollout

# Assumed order is [log_probs, values, states, actions, rewards, masks]
class RolloutStorage:
    def __init__(self):
        self.store = [[] for _ in range(NUM_ITEMS)]

    def add(self, L):
        for i in range(NUM_ITEMS):
            self.store[i].append(L[i])

    def unwind(self):
        retval = [torch.cat(self.store[i]) for i in range(NUM_ITEMS)]
        return retval

    def update_gae(self, next_v):
        self.store[4] = get_GAE(next_v, self.store[4], self.store[5], self.store[1])

    def reset(self):
        self.store = [[] for _ in range(NUM_ITEMS)]
