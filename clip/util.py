import torch
import math

# Break list or tensor into chunks
def chunk(L, sep):
    size = len(L)
    return [L[i * sep:min(size, (i+1) * sep)] for i
            in range(math.ceil(size / sep))]

# Generate indices in dataset for batch
def generate_indices(total_size, batch_size, shuffle = True):
    inds = torch.randperm(total_size) if shuffle else torch.arange(total_size)
    return chunk(inds, batch_size)

