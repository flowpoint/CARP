from encoder import TextEncoder
from model import ContrastiveModel
from dataloading import get_dataset
from util import generate_indices
import torch

# Setup model
model = ContrastiveModel(TextEncoder(), TextEncoder())
model.load_state_dict(torch.load("checkpoint.pt"))
model.cuda()

# These utility functions serve same purpose as in
# train.py
N_CTX = 512
def tok(string_batch):
    for i, _ in enumerate(string_batch):
        if len(string_batch[i]) > N_CTX:
            string_batch[i] = string_batch[i][-N_CTX:]

    return model.encA.tok(string_batch)

def get_batch_tokens(dataset, inds):
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

# Randomly run on a batch from some dataset
# If batch size is none, treats dataset as batch and does not shuffle
def get_logit(dataset, batch_size):
    dataset_size = len(dataset)
    # Randomly sample some passages and corresponding reviews
    if batch_size == -1:
        batches_inds = generate_indices(dataset_size, dataset_size, shuffle = False)
    else:
        batches_inds = generate_indices(dataset_size, batch_size)

    batch_inds = batches_inds[0]
    passages = [dataset[i][0] for i in batch_inds]
    reviews = [dataset[i][1] for i in batch_inds]
    
    pass_tokens, pass_masks, rev_tokens, rev_masks = get_batch_tokens(dataset, batch_inds)

    logits = model.getLogits([pass_tokens, pass_masks], [rev_tokens, rev_masks])

    # Now have logits, passages and reviews
    # Normalize with softmax

    conf = logits.softmax(1)
    # Find where it went wrong
    fails = []
    for i, choice in conf:
        if choice != i:
            fails.append(i)
    for i, fail in enumerate(fails):
        print("Fail " + str(i) + ":")
        print("PASSAGE: " + passages[fail])
        print("REVIEWS: (expected " + str(fail) + ")")
        for j, r in enumerate(reviews):
            print("(" + str(conf[fail][j].item()) + ") " + r)

# Print formated logit given an unordered list of passages
# and reviews
def compute_logit(passages, reviews):
    pass_tokens = tok(passages)
    rev_tokens = tok(reviews)
    pass_masks = pass_tokens['attention_mask']
    rev_masks = rev_tokens['attention_mask']
    pass_tokens = pass_tokens['input_ids']
    rev_tokens = rev_tokens['input_ids']

    logits = model.getLogits([pass_tokens, pass_masks],
                            [rev_tokens, rev_masks])
    conf = logits.softmax(1)

    for i, row in enumerate(conf):
        for j, col in enumerate(row):
            print(str(i) + "-" + str(j) + ": " + str(round(col.item(), 2)))


