import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from constants import *

class ContrastiveModel(nn.Module):
    def __init__(self, encA, encB):
        super().__init__()
        
        self.encA = encA
        self.encB = encB

        self.projA = nn.Linear(GPT_OUT, LATENT_DIM, bias = False)
        self.projB = nn.Linear(GPT_OUT, LATENT_DIM, bias = False)

        self.logit_scale = nn.Parameter(torch.ones([])) * np.log(1 / 0.07)

    def encodeX(self, x):
        return self.projA(self.encA(x))

    def encodeY(self, y):
        return self.projB(self.encB(y))

    # x, y are assumed encoding/embeddings here
    def getLogits(self, x, y):
        # normalize
        x /= x.norm(dim = -1, keepdim = True)
        y /= y.norm(dim = -1, keepdim = True)

        # cos sim on log scale
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * x_feat @ y_feat.t()

        return logits

    def forward(self, x, y):
        x = self.encodeX(x)
        y = self.encodeY(y)
        return self.getLogits(x, y)
