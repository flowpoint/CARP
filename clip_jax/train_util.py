import jax
import jax.numpy as np
import flax

from models import ContrastiveLoss, TextEncoder
from util import tree_add

# Both below run forward passes and return grad function 
# Gets gradient function over passage encoder
@jax.jit
def pass_fwd(batch, pass_encs, rev_encs):
  def fwd(pass_params, logit_scale, inds):
      pass_mb = batch[0:,:,inds,:]
      mb_pass_encs = jax.ops.index_update(pass_encs, jax.ops.index[inds], 
                                    TextEncoder().apply(pass_params, pass_mb))
      encs = np.stack((np.concatenate((mb_pass_encs)),
                        np.concatenate((rev_encs))))
      loss, _ = ContrastiveLoss().apply(logit_scale, encs)
      return loss

  return jax.grad(fwd, (0, 1))

# Gradients for review encoder
@jax.jit
def rev_fwd(batch, pass_encs, rev_encs):
  def fwd(rev_params, logit_scale, inds):
    rev_mb = batch[1,:,inds,:]
    mb_rev_encs = jax.ops_index_update(rev_encs, jax.ops.index[inds],
                                    TextEncoder().apply(rev_params, rev_mb))
    encs = np.stack((np.concatenate((pass_encs)),
                      np.concatenate((mb_rev_encs))))
    loss, _ = ContrastiveLoss().apply(logit_scale, encs)
    return loss
  
  return jax.grad(fwd, (0, 1))

# Accumulate grads given indices to microbatches
@jax.jit
def accum_grads(pass_params, rev_params, logit_scale, batch,
                pass_encs, rev_encs, mb_inds):
  # mb_inds -> NMB x MB
  NMB = len(mb_inds)

  pass_grad_func = pass_fwd(batch, pass_encs, rev_encs)
  rev_grad_func = rev_fwd(batch, pass_encs, rev_encs)

  # Get first gradients manually
  pass_grad, ls_grad1 = pass_grad_func(pass_params, logit_scale, mb_inds[0])
  rev_grad, ls_grad2 = rev_grad_func(rev_params, logit_scale, mb_inds[0])
  ls_grad = tree_add(ls_grad1, ls_grad2)

  @jax.jit
  def accum_loop(i, prev_grads):
    prev_pass_grads, prev_rev_grads, prev_ls_grads = prev_grads
    new_pass_grad, new_ls_grad1 = pass_grad_func(pass_params, logit_scale, mb_inds[i])
    new_rev_grad, new_ls_grad2 = rev_grad_func(rev_params, logit_scale, mb_inds[i])
    return (tree_add(prev_pass_grads, new_pass_grad),
            tree_add(prev_rev_grads, new_rev_grad),
            tree_add(prev_ls_grads, tree_add(new_ls_grad1,
                                             new_ls_grad2)))
  
  return jax.lax.fori_loop(1, NMB, accum_loop,
                            (pass_grad, rev_grad, ls_grads))

# Create microbatches from batch using indices
def partition_microbatches(batch, inds):
    # data: T x B x N
    batch = eo.repeat(batch, 'T B N -> 1 T B N')
    mbs = [batch[:,:,ind,:] for ind in inds]
    return np.concatenate(mbs) # -> NMB x T x MB x N
