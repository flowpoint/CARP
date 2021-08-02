import jax
import jax.numpy as np
import flax

from constants import *
from models import TextEncoder, ContrastiveLoss, FlaxTokenizers
from util import generate_indices, clip_logit, save_checkpoint
from train_util import pass_fwd, rev_fwd, accum_grads
from train_util import partition_microbatches


def train(states, dataset, evalset):
  pass_state, rev_state, ls_state = states

  tokenizer = FlaxTokenizer()
  def tok(string_batch):
    return tokenizer.tok(string_batch)
  
  # Get batch tokens and masks
  def get_batch_tokens(dataset, inds):
    batch = [dataset[ind] for ind in inds]
    pass_batch = [pair[0] for pair in batch]
    rev_batch = [pair[1] for pair in batch]

    pass_ids = tok(pass_batch) # -> [B x T x N_1]
    rev_ids = tok(rev_batch) # -> [B x T x N_2]

    return [pass_ids, rev_ids]
  
  def encode_and_val(mbs, pass_state, rev_state, ls_state):
    # mbs (microbatches) assumed NMB x T x MB x N_i
    # NMB is num microbatches
    # MB is microbatch size
    pass_mbs, rev_mbs = mbs # -> both (NMB x T x MB x N_i)

    @jax.pmap
    def get_encs(mbs, state):
      NMB = mbs.size
      encodings = np.zeros((NMB, MICROBATCH_SIZE, LATENT_DIM))
      def update_inplace(i, encodings):
        return jax.ops.index_update(encodings, jax.ops.index[i],
                                    TextEncoder().apply(state.params, mbs[i]))
      encodings = jax.lax.fori_loop(0, NMB, update_inplace, encodings)
      return np.concatenate(encodings) # -> (B x D)

    pass_state = flax.jax_utils.replicate(pass_state)
    pass_mbs = device_split(pass_mbs) # Each device gets some microbatches
    pass_encs = get_encs(pass_mbs, pass_state)
    pass_encs = device_join(pass_encs)
    pass_params = flax.jax_utils.unreplicate(pass_state)
    
    rev_state = flax.jax_utils.replicate(rev_state)
    rev_mbs = np.array_split(rev_mbs, N_DEVICES)
    rev_encs = get_encs(rev_mbs, rev_state)
    rev_encs = device_join(rev_encs)
    rev_state = flax.jax_utils.replicate(rev_state)

    encs = np.stack((pass_encs, rev_encs))

    loss, acc = ContrastiveLoss().apply(logit_scale, encs)

    # -> 2 x B x D
    return encs, loss, acc
  
  # Create microbatches from batch using indices


  dataset_size = len(dataset)
  evalset_size = len(evalset)

  total_steps = 0

  for epoch in range(EPOCHS):
    batches_inds = generate_indices(dataset_size, BATCH_SIZE)
    for batch_inds in batches_inds:
      pass_batch, rev_batch = get_batch_tokens(dataset, batch_inds)
      microbatch_inds = generate_indices(len(batch_inds), MICROBATCH_SIZE, shuffle = False)
      # -> NMB x MB (assuming microbatch_size divides batch_size) (list of arrs)
      NMB = len(microbatch_inds)

      # Split tokens and masks into these mbs
      mbs = [partition_microbatches(pass_batch, microbatch_inds),
             partition_microbatches(rev_batch, microbatch_inds)]

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

      pass_params = flax.jax_utils.replicate(pass_params)
      rev_params = flax.jax_utils.replicate(rev_params)
      logit_scale = flax.jax_utils.replicate(logit_scale)

      accum_grad_func = jax.pmap(accum_grads,
                                 static_broadcasted_argnums = [3,4,5])
      pass_grads, rev_grads, ls_grads = accum_grad_func(pass_params, rev_params,
                                                        logit_scale, batch, 
                                                        pass_encs, rev_encs)
      
      # Sum up the accumulated gradients
      pass_grads = jax.tree_map(lambda x: x.sum(0), pass_grads)
      rev_grads = jax.tree_map(lambda x: x.sum(0), rev_grads)
      ls_grads = jax.tree_map(lambda x: x.sum(0), ls_grads)
      
      pass_params = flax.jax_utils.unreplicate(pass_params)
      rev_params = flax.jax_utils.unreplicate(rev_params)
      logit_scale = flax.jax_utils.unreplicate(logit_scale)

      update_pass, pass_opt_state = pass_opt.update(pass_grads, pass_opt_state, pass_params)
      update_rev, rev_opt_state = rev_opt.update(rev_grads, rev_opt_state, rev_params)
      update_ls, ls_opt_state = ls_opt.update(ls_grads, ls_opt_state, logit_scale)

      pass_params = optax.apply_updates(pass_params, update_pass)
      rev_params = optax.apply_updates(rev_params, update_rev)
      logit_scale = optax.apply_updates(logit_scale, update_ls)

      logit_scale = clip_logit(logit_scale)

      
      # Logging (in terminal and WANDB)
      if total_steps % LOG_INTERVAL == 0:
        print("EPOCH [" + str(epoch) + "/" + str(EPOCHS) + 
              "] Batch Loss: " + str(batch_loss))
        if DO_LOG:
          wandb.log({"Loss/train": batch_loss,
                     "Acc/train": batch_acc})
      if total_steps % CHECKPOINT_INTERVAL == 0:
        print("SAVING...")
        save_checkpoint([pass_params, rev_params, logit_scale,
                         pass_opt_state, rev_opt_state, ls_opt_state])
        # Once every 10 saves, save copied backup
        if total_steps % (10 * CHECKPOINT_INTERVAL) == 0:
          save_checkpoint([pass_params, rev_params, logit_scale,
                         pass_opt_state, rev_opt_state, ls_opt_state],
                          "checkpoints", str(total_steps))



      total_steps += 1
      exit()
