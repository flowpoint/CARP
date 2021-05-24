# TRAINING
CRITIC_COEFF = 1.0 # How heavily to weight value networks loss
EPOCHS_PER_ROLLOUT = 4

# HYPERPARAMS
EPSILON = 0.2 # Clip radius for PPO loss
GAMMA = 0.99 # Discount factor
LEARNING_RATE = 2e-4
TAU = 0.95 # For GAE in util.py
