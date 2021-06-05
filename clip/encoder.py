from torch import nn
import torch

from transformers import DebertaV2Tokenizer, DebertaV2Model

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xlarge')
        self.model.to('cuda')

    def forward(self, x):
        out = self.model(x['input_ids'], x['attention_mask'], output_hidden_states = True)
        
        hidden = out[1]
        return hidden[-2].to('cuda') # 2nd Last hidden state

