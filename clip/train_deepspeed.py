import deepspeed
import torch
import torch.nn.functional as F
from torch import nn

LEARNING_RATE = 5e-5

# Calculate contrastive loss between two encodings
class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, x_embed, y_embed):
        n = x_embed.shape[0]
        x_embed = F.normalize(x_embed)
        y_embed = F.normalize(y_embed)

        logits = x_embed @ y_embed.T * torch.exp(self.logit_scale)
        labels = torch.arange(n, device = 'cuda')
    
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        acc_i = (torch.argmax(logits, dim = 1) == labels).sum()
        acc_t = (torch.argmax(logits, dim = 0) == labels).sum()

        return (loss_i + loss_t) / 2, (acc_i + acc_t) / n / 2

# Dataset assumed to be list of pairs on memory
def train(model, dataset, args = None):
    
    # Tokenizes string batch using encoder tokenizer
    # Also adds CLS tokens to end
    def tok(string_batch):
        for i, _ in enumerate(string_batch):
            if len(string_batch[i]) > N_CTX:
                string_batch[i] = string_batch[i][-N_CTX:]

        return model.encA.tok(string_batch)

    # From indices into dataset, gets batch in form of:
    # (passage tokens, passage masks, review tokens, review masks)
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
    model_engine = deepspeed.initialize(args = args, model = model,
                                        model_parameters = model.parameters(),
                                        optimizer = opt)
    
    clip_loss = CLIPLoss()

    dataset_size = len(dataset)
    torch.cuda.synchronize()
    for epoch in range(EPOCHS):
        batches_inds = generate_indices(dataset_size, BATCH_SIZE)
        for batch_inds in batches_inds:
            batch_loss = 0
            pass_tokens, pass_masks, rev_tokens, rev_masks = get_batch_tokens(batch_inds)
            microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)

            # Split tokens and masks into these microbatches
            pass_mbs = [(pass_tokens[ind], pass_masks[ind]) for ind in microbatch_inds]
            rev_mbs = [(rev_tokens[ind], rev_masks[ind]) for ind in microbatch_inds]

            # Initially get all encodings without grad
            with torch.no_grad():
                pass_encs = [model.encodeX(tokens, masks)
                        for (tokens, masks) in pass_mbs]
                
                rev_encs = [model.encodeY(tokens, masks)
                        for (tokens, masks) in rev_mbs]
                
                loss, acc = clip_loss(torch.cat(pass_encs), torch.cat(rev_encs))
            
            opt.zero_grad()

            # Encode passages in microbatches (with grad)
            for index, (tokens, masks) in enumerate(pass_mbs):
                torch.autograd.set_detect_anomaly(True)
                
                pass_tmp = pass_encs.copy()
                pass_tmp[index] = model.encodeX(tokens, masks)
                
                loss, _ = clip_loss(torch.cat(pass_tmp), torch.cat(rev_encs))
                batch_loss += loss.item()
                model_engine.backward(loss)

            # Encode reviews in microbatches (with grad)
            for index, (tokens, masks) in enumerate(rev_mbs):
                rev_tmp = rev_encs.copy()
                rev_tmp[index] = model.encodeY(tokens, masks)
                loss, _ = clip_loss(torch.cat(pass_encs), torch.cat(rev_tmp))
                batch_loss += loss.item()
                model_engine.backward(loss)

            model_engine.step()

            print("EPOCH [" + str(epoch) + "/" + str(EPOCHS) +
                  "] Batch Loss: " + str(round(batch_loss, 3)))

            print("Succesfully finished single training step!")
            exit()

from model import ContrastiveModel
from encoder import TextEncoder
from get_wp_testing import get_dataset
import util

if __name__ == "__main__":
    model = ContrastiveModel(TextEncoder(), TextEncoder())
    model.cuda()

    dataset = get_dataset()

    train(model, dataset, util.get_arguments())
    
