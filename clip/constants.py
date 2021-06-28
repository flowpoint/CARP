N_CTX = 512
EPOCHS = 10
BATCH_SIZE = 128 # Contrastive batch
MICROBATCH_SIZE = 8 # Minibatches in contrastive batch

# Size of encodings
LATENT_DIM = 2048

# info on HF model being used
import transformers
MODEL = transformers.ElectraForPreTraining
TOKENIZER = transformers.ElectraTokenizerFast
MODEL_PATH = "google/electra-large-discriminator"

# training
LR_RAMP_STEPS = 400
LR_DECAY_STEPS = 1378696/BATCH_SIZE # One full epoch
LEARNING_RATE_INIT = 5e-5
LEARNING_RATE_TARGET = 1e-6

LOG_INTERVAL = 2
CHECKPOINT_INTERVAL = 1000
VALIDATE_INTERVAL = 1000
LOAD_CHECKPOINT = False

# For dataset
VALIDATION_SIZE = 1000
