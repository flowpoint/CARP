import torch
import math
from constants import *
# Break list or tensor into chunks
def chunk(L, sep):
    size = len(L)
    return [L[i * sep:min(size, (i+1) * sep)] for i
            in range(math.ceil(size / sep))]

# Generate indices in dataset for batch
def generate_indices(total_size, batch_size, shuffle = True):
    inds = torch.randperm(total_size) if shuffle else torch.arange(total_size)
    return chunk(inds, batch_size)

# Get output dim of TextEncoder
def get_d_model(enc):
    return 1024
	#x = ["hello world", "hi world", "bla bla"]
	#y = enc(x, tokenize = True, mask_sum = False)
	#return y.shape[-1]	

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

import argparse
def get_arguments():
    parser = argparse.ArgumentParser(description = "CARP")

    parser.add_argument('--backend', type=str, default = 'nccl')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args
