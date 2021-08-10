LATENT_DIM = 2048
N_CTX = 512
TAU_PLUS = 0.1 # Constant for hard negative sampling

EPOCHS = 100
BATCH_SIZE = 512 # Contrastive Batch
MICROBATCH_SIZE = 4 # Minibatches in contrastive  batch

LOG_INTERVAL = 2
CHECKPOINT_INTERVAL = 15
VALIDATE_INTERVAL = 15

LOAD_CHECKPOINT = False
DO_LOG = False
VALIDATION_SIZE = 1000

# training
LR_RAMP_STEPS = 400
LR_DECAY_STEPS = 1378696/BATCH_SIZE # One full epoch
LEARNING_RATE = 5e-5 # "max" LR
LEARNING_RATE_MIN = 1e-6

N_DEVICES = 8

assert BATCH_SIZE % MICROBATCH_SIZE == 0
assert (BATCH_SIZE // MICROBATCH_SIZE) % N_DEVICES == 0

import jax
KEY = jax.random.PRNGKey(0)

# Get data from bucket?
# (as opposed to locally)
USE_BUCKET = True

