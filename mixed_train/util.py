import torch
import math

from constants import *

def generate_indices(total_size, batch_size, shuffle = True):
    inds = torch.randperm(total_size) if shuffle else torch.arange(total_size)
    return inds.chunk(batch_size)

# Scheduling function w/ rampup and decay
def get_scheduling_func():
    def lerp(a, b, t):
        t = min(1, t)
        t = max(0, t)
        return a + (b - a) * t

    ratio = LEARNING_RATE_TARGET / LEARNING_RATE_INIT

    return lambda step: \
        (step + 1) / LR_RAMP_STEPS if step < LR_RAMP_STEPS \
        else lerp(1, ratio, (step - LR_RAMP_STEPS) / LR_DECAY_STEPS)


def get_batch_tokens(dataset, inds):
    batch = [dataset[ind] for ind in inds]
    pass_batch = [pair[0] for pair in batch]
    rev_batch = [pair[1] for pair in batch]

    pass_tokens = tok(pass_batch)
    rev_tokens = tok(rev_batch)
    pass_masks = pass_tokens['attention_mask']
    rev_masks = rev_tokens['attention_mask']
    pass_tokens = pass_tokens['input_ids']
    rev_tokens = rev_tokens['input_ids']

    return pass_tokens, pass_masks, rev_tokens, rev_masks


