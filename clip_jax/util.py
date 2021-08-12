import jax
import jax.numpy as np
import math
import optax

from constants import *

@jax.jit
def l2norm(x):
  return x / np.linalg.norm(x, axis = 1, keepdims = True)

# Break list into chunks of sep elements
def chunk(L, sep):
  size = len(L)
  return [L[i * sep:min(size, (i+1) * sep)] for i in range(math.ceil(size / sep))]

# Generate list of indices for batches
def generate_indices(total_size, batch_size, shuffle = True):
  inds = np.arange(total_size)
  if shuffle:
    inds = jax.random.permutation(KEY, inds)
  return chunk(inds, batch_size)

# Clip parameters within state tree to below defined values
def clip_logit(ls_state):
  clamp_min = np.log(1/100)
  clamp_max = np.log(100)
  clamp_fn = lambda x: np.clip(x, clamp_min, clamp_max)
  return ls_state.replace(
      step=ls_state.step,
      params=jax.tree_map(clamp_fn, ls_state.params),
      opt_state=ls_state.opt_state
  )

# Add together two trees
@jax.jit
def tree_add(a, b):
  return jax.tree_multimap(lambda x, y: x + y, a, b)

# Get scheduling func for LR
def get_scheduling_fn():
  warmup_fn = optax.linear_schedule(
      init_value = 0.0,
      end_value = LEARNING_RATE,
      transition_steps = LR_RAMP_STEPS
  )
  decay_fn = optax.linear_schedule(
      init_value = LEARNING_RATE,
      end_value = LEARNING_RATE_MIN,
      transition_steps = LR_DECAY_STEPS - LR_RAMP_STEPS
  )
  return optax.join_schedules(
      schedules=[warmup_fn, decay_fn], boundaries = [LR_RAMP_STEPS]
  )

# Serialization
from flax import serialization
from smart_open import open

root = "gs://carp-model-storage"
# Saves checkpoint in path, name_prefix is used to differentiate checkpoints
def save_checkpoint(states, path = "", name_prefix = ""):
  for i, state in enumerate(states):
    state_bytes = serialization.to_bytes(state)
    f = open(root + path + "/state" + str(i), "wb")
    f.write(state_bytes)
    f.close()

# Same logic as above, but returns loaded states in list form
# Needs number of states being read
def load_checkpoint(n_states, path = "", name_prefix = ""):
  states = [None for _ in range(n_states)]
  for i, state in enumerate(states):
    f = open(root + path + "/state" + str(i), "rb")
    states[i] = serialization.from_bytes(state, f.read())
    f.close()

  return states

# Same as load_dataset from dataloader, but for bucket
import gcsfs
import datasets
def load_dataset_from_bucket():
    gcs = gcsfs.GCSFileSystem(project = 'carp-320015')
    dataset = datasets.load_from_disk(root + "/critiquecircle_critiques_masked_anon", fs = gcs)
    return dataset
    
# Split data on first axis across devices
# Complements replicate
def device_split(arr):
  return np.stack(np.array_split(arr, N_DEVICES))

# As above, complements unreplicate
def device_join(arr):
  return np.concatenate(arr)
