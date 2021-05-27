import torch
from torch import nn
from torch.distributions.categorical import Categorical

# Implements a PPO model based on some starting model
# Follows same process as Summarizing From Human Feedback for initialization
# Given a language model as input,
# creates separate policy/actor and critic/value networks from langauge model
class TextGenPPO(nn.Module):
    def __init__(self, start_model, ctx_len):
        super().__init__()

        self.actor = start_model.copy()
        self.critic = start_model.copy()

        self.critic_fc = nn.Linear(start_model.hidden_size, 1)

    # Get policy distribution and value of state x
    # expects state as token sequence (long tensor)
    def forward(self, x):
        act_probs = self.actor(x, return_dict = False, return_pt = True)[0]
        value = self.critic(x, return_dict = False, return_pt = True)[0]
        value = self.critic_fc(value)

        return act_probs, value

    # Use policy to generate tokens
    def generate(self, x):
        act_probs = self.actor(x, return_dict = False, return_pt = True)[0]
        dist = Categorical(act_probs)
        return dist.sample()

class BaseLinePPO(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim = 1))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        act_probs = self.actor(x)
        value = self.critic(x)

        return act_probs, value

    def act(self, x):
        act_probs = self.actor(x)
        dist = Categorical(act_probs)
        return dist.sample()

    def sample_from_probs(self, act_probs):
        dist = Categorical(act_probs)
        return dist.sample()

