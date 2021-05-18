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
    

def train(model, data_path):
    opt = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = 0.01)

    model.train()

    # TODO get data
    # From here on out, assumed "pass_batch" is a passage
    # "rev_batch" is corresponding review

    n_chunks = math.ceil(BATCH_SIZE / MICROBATCH_SIZE)

    pass_mbs = torch.chunk(pass_batch, n_chunks)
    rev_mbs = torch.chunk(rev_batch, n_chunks)

    with torch.no_grad():
        pass_encs = [model.encodeX(pass_mb) for pass_mb in pass_mbs)]
        rev_encs = [model.encodeY(rev_mb) for rev_mb in rev_mbs)]
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



