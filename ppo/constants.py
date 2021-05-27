# ENV
STATE_DIM = 4
ACT_DIM = 2

# TRAINING
CRITIC_COEFF = 0.5 # How heavily to weight value networks loss
EPOCHS_PER_ROLLOUT = 1
BATCH_SIZE = -1
MAX_GRAD_NORM = 0.5
ENT_COEFF = 0.001

# HYPERPARAMS
EPSILON = 0.2 # Clip radius for PPO loss
GAMMA = 0.98 # Discount factor
LEARNING_RATE = 1e-2
TAU = 1 # For GAE in util.py
