from torch import nn
import torch.nn.functional as F
import torch

from constants import *
import util

from transformers import AutoModel, AutoTokenizer

# For different models, hidden state is returned differently
extract_fns = {'EleutherAI/gpt-neo-1.3B' :
                (lambda out : out['hidden_states'][-1]),
                'roberta-large' : 
                (lambda out : out[0]),
                'microsoft/deberta-v2-xlarge' :
                (lambda out : out[0])}

d_models = {'EleutherAI/gpt-neo-1.3B' : 2048,
            'roberta-large' : 1024,
            'microsoft/deberta-v2-xlarge' : 1024}


class SumTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(MODEL_PATH)
        if USE_HALF: self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.d_model = util.get_d_model(self)

        # Add quote token to model and tokenizer
        self.tokenizer.add_tokens(['[quote]', '<|endoftext|>'])
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.extract_fn = extract_fns[MODEL_PATH]

    def add_eot(self, string_batch):
        return [s + "<|endoftext|>" for s in string_batch]

    def tok(self, string_batch, device="cuda"):
        return self.tokenizer(self.add_eot(string_batch),
                return_tensors = 'pt',
                padding = True).to(device)
    
    def forward(self, x, mask = None, tokenize = False, mask_sum = True, device="cuda"):
        if tokenize:
            x = self.tok(x, device)
            mask = x['attention_mask']
            x = x['input_ids']
        
        out = self.model(input_ids = x, attention_mask = mask,
                            output_hidden_states = True, return_dict = True)
        hidden = self.extract_fn(out)
        
        # Mask out pad tokens embeddings
        if mask_sum:
            emb_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
            hidden = hidden * emb_mask

        y = hidden.sum(1)
        y = F.normalize(y)
        
        return y # Sum along sequence

# Given masks returns indices of last tokens
def last_ones(t):
    # Multipliying arange by max
    # makes last non zero column have largest number in arange
    t = t * torch.arange(t.shape[1], device = 'cuda')
    # Then argmax gives index of last non zero column
    t = t.argmax(1)
    return t

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.d_model = d_models[MODEL_PATH]

        # Add quote token to model and tokenizer
        self.tokenizer.add_tokens(['[quote]', '<|endoftext|>'])
        self.tokenizer.add_special_tokens({'pad_token':'[PAD]'})

        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.extract_fn = extract_fns[MODEL_PATH]

    def add_eot(self, string_batch):
        return [s + "<|endoftext|>" for s in string_batch]

    def tok(self, string_batch, device="cuda"):
        return self.tokenizer(self.add_eot(string_batch),
                return_tensors = 'pt',
                padding = True).to(device)
    
    def forward(self, x, mask = None, tokenize = False, mask_sum = True, device="cuda"):
        if tokenize:
            x = self.tok(x, device)
            mask = x['attention_mask']
            x = x['input_ids']
        
        out = self.model(input_ids = x, attention_mask = mask,
                            output_hidden_states = True, return_dict = True)
        hidden = self.extract_fn(out) # -> B x N x D

        B, N, D = hidden.shape
        # In each mask, find last 1
        eot_inds = last_ones(mask)

        y = torch.zeros(B, D, device = 'cuda')
        for i in range(B):
            y[i] = hidden[i, eot_inds[i]] 
        # Embeddings of EOT tokens

        return y 

