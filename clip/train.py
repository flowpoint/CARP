import torch
import torch.nn.functional as F
import math

from constants import *
from util import chunk, generate_indices

# just needs logits (outer product)
def clip_loss(logits):
    # should be a square matrix
    n, _ = logits.shape
    labels = torch.arange(n, device = 'cuda')
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    acc_i = (torch.argmax(logits, dim = 1) == labels).sum()
    acc_t = (torch.argmax(logits, dim = 0) == labels).sum()

    return (loss_i + loss_t) / 2, (acc_i + acc_t) / n / 2

# NOTE: for time being dataset is just list of pairs, should prob change later
def train(model, tokenizer, dataset):
    def tok(string_batch):
        return model.encA.tok(string_batch)

    def get_batch_tokens(inds):
        batch = [dataset[ind] for ind in inds]
        pass_batch = [pair[0] for pair in batch]
        rev_batch = [pair[1] for pair in batch]

        pass_tokens = tok(pass_batch)
        rev_tokens = tok(rev_batch)
        pass_masks = pass_tokens['attention_mask']
        rev_masks = rev_tokens['attention_mask']
        pass_tokens = pass_tokens['input_ids']
        rev_tokens = rev_tokens['input_ids']

        return pass_tokens, pass_masks, rev_tokens, rev_masks

    opt = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = 0.01)

    model.train() 

    dataset_size = len(dataset)

    for epoch in range(EPOCHS):
        batches_inds = generate_indices(dataset_size, BATCH_SIZE)
        for batch_inds in batches_inds:
            pass_tokens, pass_masks, rev_tokens, rev_masks = get_batch_tokens(batch_inds)
            print("Token and mask shapes:", pass_tokens.shape, pass_masks.shape, rev_tokens.shape, rev_masks.shape)

            microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)
            with torch.no_grad():
                pass_encs = [model.encodeX(pass_tokens[ind], pass_masks[ind])
                        for ind in microbatch_inds]
                rev_encs = [model.encodeY(rev_tokens[ind], rev_masks[ind])
                        for ind in microbatch_inds]
                print("Encoding shapes:")
                print(pass_encs[0].shape, pass_encs[-1].shape)
                print(rev_encs[0].shape, rev_encs[-1].shape)
                loss, acc = clip_loss(model.getLogits(torch.cat(pass_encs),
                    torch.cat(rev_encs)))
            
            opt.zero_grad()

            for index, mb in enumerate(microbatch_inds):
                pass_tmp = pass_encs.copy()
                pass_tmp[index] = model.encodeX(pass_tokens[mb], pass_masks[mb])
                loss, _ = clip_loss(model.getLogits(torch.cat(pass_tmp),
                    torch.cat(rev_encs)))
                loss.backward()

            for index, mb in enumerate(microbatch_inds):
                rev_tmp = rev_encs.copy()
                rev_tmp[index] = model.encodeY(rev_tokens[mb], rev_masks[mb])
                loss, _ = clip_loss(model.getLogits(torch.cat(pass_encs),
                    torch.cat(rev_tmp)))
                loss.backward()

            opt.step()

            print("done!")
            exit()



