from flax.training.train_state import TrainState
import jax.numpy as np
import optax
import wandb

from constants import *
from train import train
from models import TextEncoder, ContrastiveLoss
from util import get_scheduling_fn, load_checkpoint
from dataloading import get_dataset, get_toy_dataset


optimizer = optax.adamw(
    learning_rate = get_scheduling_fn(),
    weight_decay = 0
)

# Load/create params
if LOAD_CHECKPOINT:
  pass_params, rev_params, logit_scale, pass_opt_state, rev_opt_state, ls_opt_state = load_checkpoint(6)
else:
  # Create model params
  inputs = np.ones((4, BATCH_SIZE, N_CTX))
  loss_inputs = np.ones((2, BATCH_SIZE, LATENT_DIM))
  logit_scale = ContrastiveLoss().init(KEY, loss_inputs)
  pass_params = TextEncoder().init(KEY, inputs)
  rev_params = TextEncoder().init(KEY, inputs)

  pass_params = load_pretrained(pass_params)
  rev_params = load_pretrained(rev_params)

  # Create optimizer states
  pass_opt_state = optimizer.init(pass_params) # Optimizer for passage encoder
  rev_opt_state = optimizer.init(rev_params) # Optimizer for review encoder
  ls_opt_state = optimizer.init(logit_scale) # Optimizer for logit scale

  
# Create train states
pass_state = TrainState.create(apply_fn = TextEncoder().__call__,
                                 params = pass_params,
                                 tx = optimizer)
from debug_pmap import test_pmap
test_pmap(pass_state)
exit()

rev_state = TrainState.create(apply_fn = TextEncoder().__call__,
                                 params = rev_params,
                                 tx = optimizer)
loss_state = TrainState.create(apply_fn = ContrastiveLoss().__call__,
                                 params = logit_scale,
                                 tx = optimizer)
states = [pass_state, rev_state, loss_state]

dataset, evalset = get_toy_dataset()

if DO_LOG:
  wandb.init(project='CARP-JAX', entity='shahbuland', resume = LOAD_CHECKPOINT)

print("States loaded")

