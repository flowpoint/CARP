import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import math
import wandb

from torch.cuda import amp

import encoder
from constants import *
from util import generate_indices, get_scheduling_func, get_batch_tokens


# Dataset assumed to be list of pairs in memory
def train(model, dataset, evalset):
    dataset_size = len(dataset)
    evalset_size = len(evalset)

    # Tokenize but truncate past N_CTX
    tokenizer = encoder.Tokenizer()
    def tok(string_batch):
        for i, _ in enumerate(string_batch):
            if len(string_batch[i]) > N_CTX:
                string_batch[i] = string_batch[i][-N_CTX:]
        return tokenizer.tok(string_batch)

    # Get encodings without grad
    # Also gets loss and accuracy
    def encode_and_val(pass_mbs, rev_mbs):
        with torch.no_grad():
            pass_encs = [model.encodeX(tokens, masks)
                for (tokens, masks) in pass_mbs]
            rev_encs = [model.encodeY(tokens, masks)
                for (tokens, masks) in rev_mbs]

            loss, acc = model.cLoss(torch.cat(pass_encs),
                                    torch.cat(rev_encs))
        return pass_encs, rev_encs, loss, acc

    # Optimizer, schedule and gradient scalar for amp
    opt = torch.optim.AdamW(model.parameters(),
        lr = LEARNING_RATE_INIT, weight_decay = 0)
    scheduler = LambdaLR(opt, get_scheduling_func())
    scaler = amp.GradScaler()

    if LOAD_CHECKPOINT:
        scheduler.load_state_dict(torch.load("./schedule.pt"))
        opt.load_state_dict(torch.load("./opt.pt"))

    model.train()
    
    for epoch in range(EPOCHS):
        batches_inds = generate_indices(dataset_size, BATCH_SIZE)
        for batch_inds in batches_inds:
            pass_t, pass_m, rev_t, rev_m = get_batch_tokens(dataset, batch_inds)
            microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)

            # Seperate tokens and masks into microbatches
            pass_mbs = [[pass_t[ind], pass_m[ind]] for ind in microbatch_inds]
            rev_mbs = [[rev_t[ind], rev_m[ind]] for ind in microbatch_inds]

            with amp.autocast():
                pass_encs, rev_encs, batch_loss, batch_acc = encode_and_val(pass_mbs, rev_mbs)

            opt.zero_grad()
            for index, mb in enumerate(pass_mbs):
                pass_tmp = pass_encs.copy()
                with amp.autocast():
                    pass_tmp[index] = model.encodeX(*mb)
                    loss, _ = model.cLoss(torch.cat(pass_tmp),
                                          torch.cat(rev_encs))
                scaler.scale(loss).backward()
            
            for index, mb in enumerate(rev_mbs):
                rev_tmp = rev_encs.copy()
                with amp.autocast():
                    rev_tmp[index] = model.encodeY(*mb)
                    loss, _ = model.cLoss(torch.cat(pass_encs),
                                          torch.cat(rev_encs))
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            scheduler.step()

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
            if iteration % VALIDATE_INTERVAL == 0:
                print("VALIDATING...")
                model.eval()
                val_batches_inds = generate_indices(evalset_size, BATCH_SIZE)
                val_losses, val_accs = [], []
                for batch_inds in val_batches_inds:
                    pass_t, pass_m, rev_t, rev_m = get_batch_tokens(evalset, batch_inds)
                    microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)

                    pass_mbs = [(pass_t[ind], pass_m[ind]) for ind in microbatch_inds]
                    rev_mbs = [(rev_t[ind], rev_m[ind]) for ind in microbatch_inds]
                    
                    with amp.autocast():
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


