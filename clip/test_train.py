from train import train
from model import ContrastiveModel
from encoder import TextEncoder
from get_wp_testing import get_dataset
from constants import *

model = ContrastiveModel(TextEncoder(), TextEncoder())
model.cuda()

dataset = get_dataset()

train(model, dataset)

