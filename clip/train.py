import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import math
import wandb

from constants import *
from util import chunk, generate_indices, get_scheduling_func
scaler = torch.cuda.amp.GradScaler()

# Dataset assumed to be list of pairs on memory
def train(model, dataset, evalset):
    # Tokenizes string batch using encoder tokenizer
    # Also adds CLS tokens to end
    def tok(string_batch):
        for i, _ in enumerate(string_batch):
            if len(string_batch[i]) > N_CTX:
                string_batch[i] = string_batch[i][-N_CTX:]

        return model.encA.tok(string_batch)

    # From indices into dataset, gets batch in form of:
    # (passage tokens, passage masks, review tokens, review masks)
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

    # Get encodings and validates them (gets loss and accuracy) without grad
    def encode_and_val(pass_mbs, rev_mbs):
        with torch.no_grad():
            pass_encs = [model.encodeX(tokens, masks)
                for (tokens, masks) in pass_mbs]
            
            rev_encs = [model.encodeY(tokens, masks)
                for (tokens, masks) in rev_mbs]
        
            test_loss, test_acc = model.cLoss(torch.cat(pass_encs), torch.cat(rev_encs))
        return pass_encs, rev_encs, test_loss, test_acc

    opt = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE_INIT, weight_decay = 0)
    scheduler = LambdaLR(opt, get_scheduling_func())
    if LOAD_CHECKPOINT:
        scheduler.load_state_dict(torch.load("./schedule.pt"))
        opt.load_state_dict(torch.load("./opt.pt"))
    
    model.train() 
    
    dataset_size = len(dataset)
    evalset_size = len(evalset)

    iteration = 0
    
    for epoch in range(EPOCHS):
        batches_inds = generate_indices(dataset_size, BATCH_SIZE)
        for batch_inds in batches_inds:
            pass_tokens, pass_masks, rev_tokens, rev_masks = get_batch_tokens(dataset, batch_inds)
            microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)

            # Split tokens and masks into these microbatches
            with torch.cuda.amp.autocast():
                pass_mbs = [(pass_tokens[ind], pass_masks[ind]) for ind in microbatch_inds]
                rev_mbs = [(rev_tokens[ind], rev_masks[ind]) for ind in microbatch_inds]

            # Initially get all encodings without grad
            pass_encs, rev_encs, forward_loss, forward_acc = encode_and_val(pass_mbs, rev_mbs)

            opt.zero_grad()
            # Encode passages in microbatches (with grad)
            for index, (tokens, masks) in enumerate(pass_mbs):
                pass_tmp = pass_encs.copy()
                with torch.cuda.amp.autocast():
                    pass_tmp[index] = model.encodeX(tokens, masks)
                loss, _ = model.cLoss(torch.cat(pass_tmp), torch.cat(rev_encs))
                scaler.scale(loss).backward()

            # Encode reviews in microbatches (with grad)
            for index, (tokens, masks) in enumerate(rev_mbs):
                rev_tmp = rev_encs.copy()
                with torch.cuda.amp.autocast():
                    rev_tmp[index] = model.encodeY(tokens, masks)
                loss, _ = model.cLoss(torch.cat(pass_encs), torch.cat(rev_tmp))
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()

            # Logging (in terminal and on WANDB)
            if iteration % LOG_INTERVAL == 0:
                print("EPOCH [" + str(epoch) + "/" + str(EPOCHS) +
                  "] Batch Loss: " + str(forward_loss.item()))
                if DO_LOG:
                    wandb.log({"Loss/train": forward_loss,
                            "Acc/train": forward_acc})
            # Checkpoint model and scheduler
            if iteration % CHECKPOINT_INTERVAL == 0:
                print("SAVING...")
                # Only save extra once every 20
                if iteration % (5 * CHECKPOINT_INTERVAL) == 0:
                    torch.save(model.state_dict(), "./checkpoints/" + str(iteration) \
                           + "params.pt")
                torch.save(model.state_dict(), "./params.pt")
                torch.save(scheduler.state_dict(), "./schedule.pt")
                torch.save(opt.state_dict(), "./opt.pt")
            # Run on eval set
            if (iteration+1) % VALIDATE_INTERVAL == 0:
                print("VALIDATING...")
                model.eval()
                val_batches_inds = generate_indices(evalset_size, BATCH_SIZE)
                val_losses, val_accs = [], []
                for batch_inds in val_batches_inds:
                    pass_t, pass_m, rev_t, rev_m = get_batch_tokens(evalset, batch_inds)
                    microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)

                    pass_mbs = [(pass_t[ind], pass_m[ind]) for ind in microbatch_inds]
                    rev_mbs = [(rev_t[ind], rev_m[ind]) for ind in microbatch_inds]
                    
                    _, _, val_loss, val_acc = encode_and_val(pass_mbs, rev_mbs)
                    val_losses.append(val_loss.item())
                    val_accs.append(val_acc.item())
                val_loss = sum(val_losses)/len(val_losses)
                val_acc = sum(val_accs)/len(val_accs)
                print("Validation Avg Loss: " + str(val_loss))
                print("Validation Avg Accuracy: " + str(val_acc))
                if DO_LOG:
                    wandb.log({"Loss/validation": val_loss})
                    wandb.log({"Acc/validation": val_acc})
                model.train()
            
            iteration += 1
            scheduler.step()
            model.clamp()

from model import ContrastiveModel
from encoder import TextEncoder
from dataloading import get_dataset
import util

if __name__ == "__main__":
    model = ContrastiveModel(TextEncoder(), TextEncoder())
    if LOAD_CHECKPOINT: model.load_state_dict(torch.load("./params.pt"))
    model.cuda()
    if USE_HALF: model.half()


    # Logging stuff
    if DO_LOG:
        wandb.init(project = "CARP", entity = "Shahbuland", resume = LOAD_CHECKPOINT)
        wandb.watch(model)
    
    dataset, evalset = get_dataset()
    print("data loaded")

    train(model, dataset, evalset)
