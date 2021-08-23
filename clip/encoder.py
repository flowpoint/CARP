from torch import nn
import torch.nn.functional as F
import torch

from constants import *
import util

from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(MODEL_PATH).cuda()
        if USE_HALF: self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.d_model = util.get_d_model(self)

        # Add cls token to model and tokenizer
        self.tokenizer.add_tokens(['[CLS]', '[quote]'])
        self.model.resize_token_embeddings(len(self.tokenizer))

    def add_cls(self, string_batch):
        return [s + "[CLS]" for s in string_batch]

    def tok(self, string_batch):
        return self.tokenizer(self.add_cls(string_batch),
                return_tensors = 'pt',
                padding = True).to('cuda')
    
    def forward(self, x, mask = None, tokenize = False, mask_sum = True):
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
        
        hidden = out[0]
        #layers = out[-1]
        #hidden = layers[-2]
        
        # Mask out pad tokens embeddings
        if mask_sum:
            emb_mask = mask.unsqueeze(2).repeat(1, 1, self.d_model)
            hidden = hidden * emb_mask

        y = hidden.sum(1)
        y = F.normalize(y)
        
        return y # Sum along sequence
