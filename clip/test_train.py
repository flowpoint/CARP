from train import train
from model import ContrastiveModel
from encoder import TextEncoder
from get_wp_testing import get_dataset
from transformers import DebertaV2Tokenizer

model = ContrastiveModel(TextEncoder(), TextEncoder())
dataset = get_dataset()
tokenizer =  DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge')

train(model, tokenizer, dataset)
print("done!")

