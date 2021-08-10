import jax
import jax.numpy as np
import flax

from models import ContrastiveLoss, TextEncoder
from util import tree_add

# Both below run forward passes and return grad function 
# Gets gradient function over passage encoder
def pass_fwd():
  def fwd(pass_params, logit_scale, data):
      batch, pass_encs, rev_encs, inds = data
      # batch is 4 x B x N
      # only care about microbatch, which should be 4 x MB x N
      pass_mb = batch[0][:,inds,:]
      mb_pass_encs = jax.ops.index_update(pass_encs, jax.ops.index[inds], 
                                    TextEncoder().apply(pass_params, pass_mb))
      encs = np.stack((np.concatenate((mb_pass_encs)),
                        np.concatenate((rev_encs))))
      loss, _ = ContrastiveLoss().apply(logit_scale, encs)
      return loss

  return jax.grad(fwd, (0, 1))

# Gradients for review encoder
def rev_fwd():
  def fwd(rev_params, logit_scale, data):
    batch, pass_encs, rev_encs, inds = data
    rev_mb = batch[1][:,inds,:]
    mb_rev_encs = jax.ops.index_update(rev_encs, jax.ops.index[inds],
                                    TextEncoder().apply(rev_params, rev_mb))
    encs = np.stack((np.concatenate((pass_encs)),
                      np.concatenate((mb_rev_encs))))
    loss, _ = ContrastiveLoss().apply(logit_scale, encs)
    return loss
  
  return jax.grad(fwd, (0, 1))

# Accumulate grads given indices to microbatches
def accum_grads_pass(enc_state, ls_state, batch,
                pass_encs, rev_encs, mb_inds):
  # mb_inds -> NMB x MB
  NMB = len(mb_inds)

  grad_fn = pass_fwd()

  # Get first gradients manually
  enc_grad, ls_grad = grad_fn(enc_state.params, ls_state.params, 
                              (batch, pass_encs, rev_encs, mb_inds[0]))

  def accum_loop(i, inp): #inp: prev_enc_grad, prev_ls_grad, grad_fn, state
    prev_enc_grads, prev_ls_grads = inp
    new_enc_grad, new_ls_grad = grad_fn(enc_state.params, ls_state.params,
                                        (batch, pass_encs, rev_encs, mb_inds[i]))
    return (tree_add(prev_enc_grads, new_enc_grad),
            tree_add(prev_ls_grads, new_ls_grad))
  
  enc_grad, ls_grad = jax.lax.fori_loop(1, NMB, accum_loop,
                                         (enc_grad, ls_grad))
  return enc_grad, ls_grad

def accum_grads_rev(enc_state, ls_state, batch,
                pass_encs, rev_encs, mb_inds):
  # mb_inds -> NMB x MB
  NMB = len(mb_inds)

  grad_fn = rev_fwd()

  # Get first gradients manually
  enc_grad, ls_grad = grad_fn(enc_state.params, ls_state.params, 
                              (batch, pass_encs, rev_encs, mb_inds[0]))

  def accum_loop(i, inp): #inp: prev_enc_grad, prev_ls_grad, grad_fn, state
    prev_enc_grads, prev_ls_grads = inp
    new_enc_grad, new_ls_grad = grad_fn(enc_state.params, ls_state.params,
                                        (batch, pass_encs, rev_encs, mb_inds[i]))
    return (tree_add(prev_enc_grads, new_enc_grad),
            tree_add(prev_ls_grads, new_ls_grad))
  
  enc_grad, ls_grad = jax.lax.fori_loop(1, NMB, accum_loop,
                                         (enc_grad, ls_grad))
  return enc_grad, ls_grad
