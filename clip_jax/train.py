import jax
import jax.numpy as np
import flax
import einops as eo
import wandb
from jax.interpreters import xla

from constants import *
from models import TextEncoder, ContrastiveLoss, FlaxTokenizer
from util import generate_indices, clip_logit, save_checkpoint
from util import device_split, device_join, tree_add
from train_util import pass_fwd, rev_fwd, accum_grads_pass, accum_grads_rev

# Dimensions:
# B: Batch
# MB: Microbatch
# NMB: All microbatches
# N_1: sequence length for passages
# N_2: sequence length for reviews
# D: Latent dim
# T: Different tokenizer outputs

# Take model states (as shown in tuple below), dataset and validation set
def train(states, dataset, evalset):
  pass_state, rev_state, ls_state = states

  tokenizer = FlaxTokenizer()
  def tok(string_batch):
    return tokenizer.tok(string_batch)
  
  # Get tokenizer output over batch specified by inds
  def get_batch_tokens(dataset, inds):
    batch = [dataset[ind] for ind in inds]
    pass_batch = [pair[0] for pair in batch]
    rev_batch = [pair[1] for pair in batch]

    pass_ids = tok(pass_batch) # -> [B x T x N_1]
    rev_ids = tok(rev_batch) # -> [B x T x N_2]

    return [pass_ids, rev_ids]
  
  # Without gradients, encodes minibatches using model
  # Returns encodings, loss and accuracy 
  # Assumes states are replicated
  def encode_and_val(mbs, pass_state, rev_state, ls_state):
    # mbs (microbatches) assumed NMB x T x MB x N_i
    # NMB is num microbatches
    # MB is microbatch size
    pass_mbs, rev_mbs = mbs # -> both (NMB x T x MB x N_i)

    # Assuming mbs is a list, iterates through it and encodes all microbatches
    def get_encs(mbs, state):
      NMB = len(mbs)
      encodings = np.zeros((NMB, MICROBATCH_SIZE, LATENT_DIM))
      def update_inplace(i, encodings):
        return jax.ops.index_update(encodings, jax.ops.index[i],
                                    TextEncoder().apply(state.params, mbs[i]))
      encodings = jax.lax.fori_loop(0, NMB, update_inplace, encodings)
      return np.concatenate(encodings) # -> (B x D)

    # Each device encodes share of microbatches
    get_encs = jax.pmap(get_encs)

    # Split passage microbatches between devices and encode
    pass_mbs = device_split(pass_mbs)
    pass_encs = get_encs(pass_mbs, pass_state)
    pass_encs = device_join(pass_encs)
    
    rev_mbs = device_split(rev_mbs)
    rev_encs = get_encs(rev_mbs, rev_state)
    rev_encs = device_join(rev_encs)

    encs = np.stack((pass_encs, rev_encs))

    # Unreplicate logit scale and calculate overall loss for the batch
    ls_unrp = flax.jax_utils.unreplicate(ls_state)
    loss, acc = ContrastiveLoss().apply(ls_unrp.params, encs)

    # -> 2 x B x D
    return encs, loss, acc
  
  # Create microbatches from batch using indices
  def partition_microbatches(batch, inds):
    # data: T x B x N
    batch = eo.repeat(batch, 'T B N -> 1 T B N')
    mbs = [batch[:,:,ind,:] for ind in inds]
    return np.concatenate(mbs) # -> NMB x T x MB x N

  # Performs train step with model states, batch, already computed encodings 
  # and microbatch indices
  def train_step(pass_state, rev_state, ls_state, batch, pass_encs, rev_encs, microbatch_inds):
    # Get gradients for passage encoder
    pass_grads, ls_grads1 = accum_grads_pass(pass_state, ls_state,
                                            batch, pass_encs, rev_encs, microbatch_inds)
    pass_grads = jax.lax.pmean(pass_grads, "batch")

    # Gradients for review encoder
    rev_grads, ls_grads2 = accum_grads_rev(rev_state, ls_state,
                                            batch, pass_encs, rev_encs, microbatch_inds)
    rev_grads = jax.lax.pmean(rev_grads, "batch")

    #ls_grads1 = tree_add(ls_grads1, ls_grads2) # Combine gradients for logit scale
    ls_grads1 = jax.lax.pmean(ls_grads1, "batch")

    new_pass = pass_state.apply_gradients(grads=pass_grads)
    new_rev = rev_state.apply_gradients(grads=rev_grads)
    new_ls = ls_state.apply_gradients(grads=ls_grads1)

    return [new_pass, new_rev, new_ls]

  dataset_size = len(dataset)
  evalset_size = len(evalset)

  total_steps = 0

  # Because splits all depend on divisible sizes,
  # need to skip a batch that isn't fixed size
  skip_last_batch = dataset_size % BATCH_SIZE != 0
  val_skip_last_batch = evalset_size % BATCH_SIZE != 0

  # Replicate models
  pass_state = flax.jax_utils.replicate(pass_state)
  rev_state = flax.jax_utils.replicate(rev_state)
  ls_state = flax.jax_utils.replicate(ls_state)

  for epoch in range(EPOCHS):
    batches_inds = generate_indices(dataset_size, BATCH_SIZE)
    if skip_last_batch: batches_inds = batches_inds[:-1]
    for batch_inds in batches_inds:
      pass_batch, rev_batch = get_batch_tokens(dataset, batch_inds)
      microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)
      # -> NMB x MB (assuming microbatch_size divides batch_size) (list of arrs)
      NMB = len(microbatch_inds)

      # Split tokens and masks into these mbs
      mbs = [partition_microbatches(pass_batch, microbatch_inds),
             partition_microbatches(rev_batch, microbatch_inds)]
        
      # pre-compute encodings and get overall loss/acc over batch
      encs, batch_loss, batch_acc = encode_and_val(mbs, pass_state,
                                                   rev_state, ls_state)
      pass_encs, rev_encs = encs
      # Have to break these encodings both back into microbatches
      pass_encs = np.stack(np.array_split(pass_encs, NMB))
      rev_encs = np.stack(np.array_split(rev_encs, NMB))
      # -> NMB x MB x D

      # NMB x MB samples
      # Split microbatch inds across devices
      microbatch_inds = device_split(np.stack(microbatch_inds))

      # Old model states are donated, since they will be replaced
      p_train_step = jax.pmap(train_step, "batch",
                                static_broadcasted_argnums=[3,4,5],
                                donate_argnums=[0,1,2,6])
      pass_state, rev_state, ls_state = p_train_step(pass_state, rev_state, ls_state,
                                                    (pass_batch, rev_batch), pass_encs, rev_encs,
                                                    microbatch_inds)

      ls_state = clip_logit(ls_state)

      
      # Logging (in terminal and WANDB)
      if total_steps % LOG_INTERVAL == 0:
        print("EPOCH [" + str(epoch) + "/" + str(EPOCHS) + 
              "] Batch Loss: " + str(batch_loss) + ", Batch Acc: " + str(batch_acc))
        if DO_LOG:
          wandb.log({"Loss/train": batch_loss,
                     "Acc/train": batch_acc})
          
      if SAVE_CHECKPOINTS and (total_steps % CHECKPOINT_INTERVAL == 0):
        print("SAVING...")
        unrp = flax.jax_utils.unreplicate
        states = [unrp(pass_state), unrp(rev_state), unrp(ls_state)]
        save_checkpoint(states)
        # Once every 10 saves, save copied backup
        if total_steps % (10 * CHECKPOINT_INTERVAL) == 0:
          save_checkpoint(states,
                          "/checkpoints", str(total_steps))

      if DO_VALIDATE and (total_steps % VALIDATE_INTERVAL == 0):
        print("VALIDATING...")
        val_batches_inds = generate_indices(evalset_size, BATCH_SIZE)
        if val_skip_last_batch: val_batches_inds = val_batches_inds[:-1]
        val_losses, val_accs = [], []
        for batch_inds in val_batches_inds:
          pass_batch, rev_batch = get_batch_tokens(evalset, batch_inds)
          microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)

          mbs = [partition_microbatches(pass_batch, microbatch_inds),
                 partition_microbatches(rev_batch, microbatch_inds)]
          
          _, val_loss, val_acc = encode_and_val(mbs, pass_state, rev_state, ls_state)
          val_losses.append(val_loss)
          val_accs.append(val_acc)

        val_loss = sum(val_losses)/len(val_losses)
        val_acc = sum(val_accs)/len(val_accs)

        print("Validation Avg Loss: " + str(val_loss))
        print("Validation Avg Accuracy: " + str(val_acc))

        if DO_LOG:
          wandb.log({"Loss/validation": val_loss})
          wandb.log({"Acc/validation": val_acc})

      total_steps += 1

