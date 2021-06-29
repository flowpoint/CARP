import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from constants import *

class ContrastiveModel(nn.Module):
    def __init__(self, encA, encB):
        super().__init__()
        
        self.encA = encA
        self.encB = encB

        self.projA = nn.Linear(self.encA.d_model, LATENT_DIM, bias = False)
        self.projB = nn.Linear(self.encB.d_model, LATENT_DIM, bias = False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clamp_min = math.log(1/100)
        self.clamp_max = math.log(100)

    def clamp(self):
        with torch.no_grad():
            self.logit_scale.clamp(self.clamp_min, self.clamp_max)

    def encodeX(self, x, masks = None):
        x = self.encA(x, masks)
        return self.projA(x)

    def encodeY(self, y, masks = None):
        y = self.encB(y, masks)
        return self.projB(y)

    # Calculate contrastive loss between embedding groups
    # x, y are assumed encoding/embeddings here
    def cLoss(self, x, y):
        n = x.shape[0]
        # normalize
        x = F.normalize(x)
        y = F.normalize(y)

        logits = x @ y.T * self.logit_scale.exp()
        labels = torch.arange(n, device = 'cuda')

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        acc_i = (torch.argmax(logits, dim = 1) == labels).sum()
        acc_t = (torch.argmax(logits, dim = 0) == labels).sum()

        return (loss_i + loss_t) / 2, (acc_i + acc_t) / n / 2

    def forward(self, x, y):
        x = self.encodeX(x)
        y = self.encodeY(y)
        return self.getLogits(x, y)
