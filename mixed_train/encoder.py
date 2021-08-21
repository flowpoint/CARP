from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn

from constants import *

class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.tokenizer.add_tokens(["[CLS]", "[quote]"])

    def add_cls(self, string_batch):
        return [s + "[CLS]" for s in string_batch]

    def tok(self, string_batch):
        return self.tokenizer(self.add_cls(string_batch),
            return_tensors = 'pt',
            padding = True)

# Different models output hidden states in different ways
# Useful to keep map of how to extract from model output
extract_fns = {'roberta-large': lambda out: out[0]}

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained(MODEL_PATH)
        self.embed = extract_fns[MODEL_PATH]

    def forward(self, x, mask = None):
        out = self.model(x, mask, output_hidden_states = True, returns_dict = True)
        hidden = self.embed(out)

        if mask_sum:
            emb_mask = mask.unsqueeze(2).repeat(1, 1, D_MODEL)
            hidden = hidden * emb_mask

        y = hidden.sum(1)
        y = F.normalize(y)

        return y
        
