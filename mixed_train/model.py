import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from constants import *
from encoder import TextEncoder

class ContrastiveModel(nn.Module):
    def __init__(self, encX = TextEncoder(), encY = TextEncoder()):
        super().__init__()

        self.encX = encX
        self.encY = encY

        self.projX = nn.Linear(D_MODEL, LATENT_DIM)
        self.projY = nn.Linear(D_MODEL, LATENT_DIM)

        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clamp_min = np.log(1/100)
        self.clamp_max = np.log(100)

    def clamp(self):
        with torch.no_grad():
            self.temp.clamp(self.clamp_min, self.clamp_max)

    def encodeX(self, x, masks = None):
        x = self.encX(x, masks)
        return self.projX(x)

    def encodeY(self, y, masks = None):
        y = self.encY(y, masks)
        return self.projY(y)

    def cLoss(self, x, y):
        n = x.shape[0]

        x = F.normalize(x)
        y = F.normalize(y)

        logits = x @ y.T * self.temp.exp()
        labels = torch.arange(n, device = 'cuda')

        loss_x = F.cross_entropy(logits, labels)
        loss_y = F.cross_entropy(logits.T, labels)
        acc_x = (torch.argmax(logits, dim = 1) == labels).sum()
        acc_y = (torch.argmax(logits, dim = 0) == labels).sum()

        return (loss_x + loss_y) / 2, (acc_x + acc_y) / n / 2

    def getLogits(self, x, y):
        x = self.encodeX(*x)
        y = self.encodeY(*y)

        x = F.normalize(x)
        y = F.normalize(y)

        logits = x @ y.T * self.temp.exp()
        return logits

    def forward(self, x, y):
        return self.getLogits(x, y)
