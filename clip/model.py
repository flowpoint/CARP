import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from constants import *

# Project encoder output to latent space
class Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(PROJ_DROPOUT)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        projected = self.proj(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)

class ContrastiveModel(nn.Module):
    def __init__(self, encA, encB):
        super().__init__()
        
        self.encA = encA
        self.encB = encB

        if LINEAR_PROJECTION:
            self.projA = nn.Linear(self.encA.d_model, LATENT_DIM, bias = False)
            self.projB = nn.Linear(self.encB.d_model, LATENT_DIM, bias = False)
        else:
            self.projA = Projection(self.encA.d_model, LATENT_DIM)
            self.projB = Projection(self.encB.d_model, LATENT_DIM)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clamp_min = math.log(1/100)
        self.clamp_max = math.log(100)

    def clamp(self):
        with torch.no_grad():
            self.logit_scale.clamp(self.clamp_min, self.clamp_max)

    def encodeX(self, x, masks = None, device='cuda'):
        x = self.encA(x, masks, device=device)
        return self.projA(x)

    def encodeY(self, y, masks = None, device='cuda'):
        y = self.encB(y, masks, device=device)
        return self.projB(y)

    # Calculate contrastive loss between embedding groups
    # x, y are assumed encoding/embeddings here
    def cLoss(self, x, y, device='cuda'):
        n = x.shape[0]
        # normalize
        x = F.normalize(x)
        y = F.normalize(y)

        logits = x @ y.T * self.logit_scale.exp()
        labels = torch.arange(n, device = device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        acc_i = (torch.argmax(logits, dim = 1) == labels).sum()
        acc_t = (torch.argmax(logits, dim = 0) == labels).sum()

        return (loss_i + loss_t) / 2, (acc_i + acc_t) / n / 2

    def forward(self, x, y, device="cuda"):
        x = self.encodeX(x, device=device)
        y = self.encodeY(y, device=device)
        return self.getLogits(x, y)
