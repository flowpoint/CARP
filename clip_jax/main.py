from flax.training.train_state import TrainState
import jax.numpy as np
import optax
import wandb

from constants import *
from train import train
from models import TextEncoder, ContrastiveLoss, load_pretrained

from util import get_scheduling_fn, load_checkpoint
from dataloading import get_dataset, get_toy_dataset


optimizer = optax.adamw(
    learning_rate = get_scheduling_fn(),
    weight_decay = 0
)

def make_train_state(state_dict, apply_fn):
  return TrainState.create(apply_fn = apply_fn,
                           params = state_dict,
                           tx = optimizer)

# Load/create params
if LOAD_CHECKPOINT:
  pass_state, rev_state, ls_state = load_checkpoint(3)
else:
  # Create model params
  inputs = np.ones((TOKENIZER_OUTPUTS, MICROBATCH_SIZE, N_CTX))
  loss_inputs = np.ones((2, MICROBATCH_SIZE, LATENT_DIM))
  print("Initializing models...")
  logit_scale = ContrastiveLoss().init(KEY, loss_inputs)
  pass_params = TextEncoder().init(KEY, inputs)
  rev_params = TextEncoder().init(KEY, inputs)

  print("Attempting to load pretrained models for finetuning...")
  pass_params = load_pretrained(pass_params)
  rev_params = load_pretrained(rev_params)

  # Create train states
  print("Creating train states...")
  pass_state = make_train_state(pass_params, TextEncoder().__call__)
  rev_state = make_train_state(rev_params, TextEncoder().__call__)
  ls_state = make_train_state(logit_scale, ContrastiveLoss().__call__)

states = [pass_state, rev_state, ls_state]

print("Loading dataset...")
dataset, evalset = get_dataset()
print("Dataset loaded (size: " + str(len(dataset)) + ")")

if DO_LOG:
  wandb.init(project='CARP-JAX', entity='shahbuland', resume = LOAD_CHECKPOINT)

train(states, dataset, evalset)
