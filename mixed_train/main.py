import wandb
import torch

from constants import *
from model import ContrastiveModel
from dataloading import get_dataset
import util
from train import train

if __name__ == "__main__":
    model = ContrastiveModel().cuda()
    if LOAD_CHECKPOINT: model.load_state_dict(torch.load("./params.pt"))

    # Logging stuff
    if DO_LOG:
        wandb.init(project = "CARP", entity = "Shahbuland", resume = LOAD_CHECKPOINT)
        wandb.watch(model)

    dataset, evalset = get_dataset()
    print("data loaded")

    train(model, dataset, evalset)
