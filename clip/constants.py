N_CTX = 512
EPOCHS = 10
BATCH_SIZE = 2048 # Contrastive batch
MICROBATCH_SIZE = 32 # Minibatches in contrastive batch

# Size of encodings
LATENT_DIM = 2048
PROJ_DROPOUT = 0.1
LINEAR_PROJECTION = True # Uses more complex projection head if false

# info on HF model being used
MODEL_PATH = "klue/roberta-small"
MODEL_TYPE = "mlm" # ar or mlm, determines encoder type
# training
LR_RAMP_STEPS = 400
LR_DECAY_STEPS = (1378696/BATCH_SIZE)*5 # One full epoch
LEARNING_RATE_INIT = 5e-5
LEARNING_RATE_TARGET = 3e-6

GRAD_CLIP = 1.0 # What to clip grad norms to (set to -1 for no clip)

LOG_INTERVAL = 2
CHECKPOINT_INTERVAL = 50
VALIDATE_INTERVAL = 50
LOAD_CHECKPOINT = False
USE_HALF = False
DO_LOG = True # Log to WANDB?
# For dataset
VALIDATION_SIZE = 1000
USE_BUCKET = False

