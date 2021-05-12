from torch import nn
import torch

from transformers import DebertaV2Tokenizer, DebertaV2Model

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xlarge')
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge')
        self.model.to('cuda')

    def tokenize(self, x):
        y = self.tokenizer.encode(x)
        y = torch.tensor([y], device = 'cuda')
        return y

    def forward(self, x):
        out = self.model(x, return_dict = False, output_hidden_states = True)
        hidden = out[2]
        return hidden[-2] # 2nd Last hidden state

