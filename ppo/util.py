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

NUM_ITEMS = 5 # Different items to store in rollout

class RolloutStorage:
    def __init__(self, max_size):
        self.store = [[] for _ in range(NUM_ITEMS)]

    def add(self, L):
        for i in range(NUM_ITEMS):
            self.store[i].append(L[i])

    def unwind(self):
        retval = [torch.cat(self.store[i]) for i in range(ROLLOUT_ITEMS)]
        return retval

    def update_gae(self, next_v):
        self.store[4] = get_GAE(next_v, self.store[4], self.store[5], self.store[1])

    def reset(self):
        self.store = [[] for _ in range(NUM_ITEMS)]
