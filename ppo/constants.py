# TRAINING
CRITIC_COEFF = 1.0 # How heavily to weight value networks loss
EPOCHS_PER_ROLLOUT = 5
BATCH_SIZE = -1
MAX_GRAD_NORM = 5

# HYPERPARAMS
EPSILON = 0.2 # Clip radius for PPO loss
GAMMA = 0.95 # Discount factor
LEARNING_RATE = 1e-3
TAU = 0.95 # For GAE in util.py
