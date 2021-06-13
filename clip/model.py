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

        self.projA = nn.Linear(self.encA.d_model, LATENT_DIM, bias = False)
        self.projB = nn.Linear(self.encB.d_model, LATENT_DIM, bias = False)

        self.logit_scale = nn.Parameter(torch.ones([])) * np.log(1 / 0.07)

    def encodeX(self, x, masks = None):
        x = self.encA(x, masks)
        return self.projA(x)

    def encodeY(self, y, masks = None):
        y = self.encB(y, masks)
        return self.projB(y)

    # x, y are assumed encoding/embeddings here
    def getLogits(self, x, y):
        print("Logit input shapes:")
        print(x.shape)
        print(y.shape)
        # normalize
        x = F.normalize(x)
        y = F.normalize(y)

        # cos sim on log scale
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * x @ y.T

        return logits

    def forward(self, x, y):
        x = self.encodeX(x)
        y = self.encodeY(y)
        return self.getLogits(x, y)
