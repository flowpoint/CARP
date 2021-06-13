N_CTX = 512
EPOCHS = 10
BATCH_SIZE = 4
MICROBATCH_SIZE = 2 # If batch size is too much for mem

# Size of encodings
LATENT_DIM = 2048

# info on HF model being used
import transformers
MODEL = transformers.ElectraForPreTraining
TOKENIZER = transformers.ElectraTokenizerFast
MODEL_PATH = "google/electra-large-discriminator"

# training
LEARNING_RATE = 1e-4
