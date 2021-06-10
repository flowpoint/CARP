from torch import nn
import torch

from transformers import DebertaV2Tokenizer, DebertaV2Model

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xlarge').cuda()
        
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge')

    def add_cls(self, string_batch):
        return [s + "[CLS]" for s in string_batch]

    def tok(self, string_batch):
        return self.tokenizer(self.add_cls(string_batch),
                return_tensors = 'pt',
                padding = True).to('cuda')

    def forward(self, x, mask = None, tokenize = False):
        if tokenize:
            x = self.tok(x)
            mask = x['attention_mask']
            x = x['input_ids']

        
        out = self.model(x, mask, output_hidden_states = True, return_dict = True)
        
        # out is a tuple of (model output, tuple)
        # the second tuple is all layers
        # in this second tuple, last elem is model output
        # we take second last hidden -> third last layer
        # size is always [batch, seq, 1536]

        layers = out[1]
        hidden = layers[-2]
        hidden = hidden[:, -1, :] # Get hidden state corresponding to the CLS token
        return hidden
