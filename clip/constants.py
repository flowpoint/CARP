N_CTX = 512
EPOCHS = 10
BATCH_SIZE = 2048 # Contrastive batch
MICROBATCH_SIZE = 14 # Minibatches in contrastive batch

# Size of encodings
LATENT_DIM = 2048

# info on HF model being used
MODEL_PATH = "roberta-large"

# training
LR_RAMP_STEPS = 400
LR_DECAY_STEPS = (1378696/BATCH_SIZE)*5 # One full epoch
LEARNING_RATE_INIT = 5e-5
LEARNING_RATE_TARGET = 3e-6

LOG_INTERVAL = 2
CHECKPOINT_INTERVAL = 50
VALIDATE_INTERVAL = 50
LOAD_CHECKPOINT = False
USE_HALF = False
DO_LOG = True # Log to WANDB?
# For dataset
VALIDATION_SIZE = 1000
USE_BUCKET = False

