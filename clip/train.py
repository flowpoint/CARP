import torch
import torch.nn.functional as F
import math

from constants import *

# just needs logits (outer product)
def clip_loss(logits, dataset):
    # should be a square matrix
    n, _ = logits.shape
    labels = torch.arange(n, device = 'cuda')
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    acc_i = (torch.argmax(logits, dim = 1) == labels).sum()
    acc_t = (torch.argmax(logits, dim = 0) == labels).sum()

    return (loss_i + loss_t) / 2, (acc_i + acc_t) / n / 2

# Generate indices in dataset for batch
def generate_indices(total_size, batch_size):
    inds = torch.randperm(total_size)
    return [inds[i * batch_size:(i+1) * batch_size] for i
            in range(0, total_size // batch_size)]

def chunk(L, sep):
    size = len(L)
    return [L[i * sep:(i+1) * sep] for i in range(0, size // sep)]

# NOTE: for time being dataset is just list of pairs, should prob change later
def train(model, tokenizer, dataset):
    def add_cls(string_batch):
        return [s + "[CLS]" for s in string_batch]

    def tok(string_batch):
        return tokenizer(add_cls(string_batch), return_tensors = 'pt',
                padding = True).to('cuda')

    opt = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = 0.01)

    model.train() 

    dataset_size = len(dataset)

    for epoch in range(EPOCHS):
        batch_inds = generate_indices(dataset_size, BATCH_SIZE)
        for batch_ind in batch_inds:
            batch = [dataset[ind] for ind in batch_ind]

            pass_batch = [pair[0] for pair in batch]
            rev_batch = [pair[1] for pair in batch]
            
            # batches to microbatch
            pass_mbs = chunk(pass_batch, MICROBATCH_SIZE)
            rev_mbs = chunk(rev_batch, MICROBATCH_SIZE)

            pass_mbs = [tok(pass_mb) for pass_mb in pass_mbs]
            rev_mbs = [tok(rev_mb) for rev_mb in rev_mbs]

            with torch.no_grad():
                pass_encs = [model.encodeX(pass_mb) for pass_mb in pass_mbs]
                rev_encs = [model.encodeY(rev_mb) for rev_mb in rev_mbs]
                loss, acc = clip_loss(model.getLogits(torch.cat(pass_encs),
                    torch.cat(rev_encs)))
            
            opt.zero_grad()

            for index, mb in enumerate(pass_mbs):
                pass_tmp = pass_encs.copy()
                pass_tmp[index] = model.encodeX(mb)
                loss, _ = clip_loss(model.getLogits(torch.cat(pass_tmp),
                    torch.cat(rev_encs)))
                loss.backward()

            for index, mb in enumerate(rev_envs):
                rev_tmp = rev_encs.copy()
                rev_tmp[index] = model.encodeY(mb)
                loss, _ = clip_loss(model.getLogits(torch.cat(pass_encs),
                    torch.cat(rev_tmp)))
                loss.backward()

            opt.step()

            print("done!")
            exit()



