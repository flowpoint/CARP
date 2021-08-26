N_CTX = 512
EPOCHS = 10
BATCH_SIZE = 2040 # Contrastive batch
MICROBATCH_SIZE = 14 # Minibatches in contrastive batch

# Size of encodings
LATENT_DIM = 2048
PROJ_DROPOUT = 0.1
LINEAR_PROJECTION = True # Uses more complex projection head if false

# info on HF model being used
MODEL_PATH = "EleutherAI/gpt-neo-2.7B"

# training
LR_RAMP_STEPS = 400
LR_DECAY_STEPS = 2*(1378696/BATCH_SIZE) # inside is number of steps in full epoch
LEARNING_RATE_INIT = 5e-5
LEARNING_RATE_TARGET = 1e-6

GRAD_CLIP = 1.0 # What to clip grad norms to (set to -1 for no clip)

LOG_INTERVAL = 2
CHECKPOINT_INTERVAL = 15
VALIDATE_INTERVAL = 50
LOAD_CHECKPOINT = False
USE_HALF = False
DO_LOG = False # Log to WANDB?
# For dataset
VALIDATION_SIZE = 1000
USE_BUCKET = False

