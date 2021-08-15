import jax
import jax.numpy as np
import einops as eo
import flax
import wandb

from functools import partial
from jax import pmap, vmap
from jax.lax import psum
import optax
from flax.training.train_state import TrainState

from constants import *
from models import FlaxTokenizer
from models import ContrastiveLoss2 as ContrastiveLoss
from models import TextEncoder2 as TextEncoder
import models
import util
import dataloading

# ==== INIT MODELS AND DATA ====

rng = jax.random.PRNGKey(0)
rng1, rng2 = jax.random.split(rng, 2)

optimizer = optax.adamw(
    learning_rate = util.get_scheduling_fn(),
    weight_decay = 0
)

def make_train_state(state_dict, apply_fn):
  return TrainState.create(apply_fn = apply_fn,
                           params = state_dict,
                           tx = optimizer)

print("Initialize Models")
if LOAD_CHECKPOINT:
    pass_state, rev_state, ls_state = util.load_checkpoint(3)
else:
    inputs = np.ones((2, TOKENIZER_OUTPUTS, N_CTX))
    loss_inputs = np.ones((BATCH_SIZE, BATCH_SIZE))
    
    logit_scale = ContrastiveLoss().init(rng, loss_inputs, np.arange(BATCH_SIZE))
    pass_params = TextEncoder().init(rng1, inputs)
    rev_params = TextEncoder().init(rng2, inputs)

    print("Attempting to load pretrained models for finetuning...")
    pass_params = models.load_pretrained(pass_params)
    rev_params = models.load_pretrained(rev_params)

    # Create train states
    print("Creating train states...")
    pass_state = make_train_state(pass_params, TextEncoder().__call__)
    rev_state = make_train_state(rev_params, TextEncoder().__call__)
    ls_state = make_train_state(logit_scale, ContrastiveLoss().__call__)

states = [pass_state, rev_state, ls_state]

dataset, evalset = dataloading.get_dataset()

if DO_LOG:
    wandb.init(project="CARP-JAX", entity = 'shahbuland', resume = False)

MICROBATCHING_FACTOR = (BATCH_SIZE // MICROBATCH_SIZE)
CORES = N_DEVICES

def validate_batch(passages, reviews):
    passages = eo.rearrange(passages, '(cores examples) t tokens -> corex examples t tokens', cores=CORES)
    reviews = eo.rearrange(reviews, '(cores examples) t tokens -> corex examples t tokens', cores=CORES)

    pass_encs = pmap(TextEncoder().apply)(pass_state.params, passages)
    pass_encs = eo.rearrance(pass_encs, "cores examples features -> (cores examples) features")
    pass_encs = util.l2norm(pass_encs)

    rev_encs = pmap(TextEncoder().apply)(rev_state.params, reviews)
    rev_encs = eo.rearrange(rev_encs, "cores examples features -> (cores examples) features")
    rev_encs = util.l2norm(rev_encs)

    sim = np.einsum("i d, j d -> i j", pass_encs, rev_encs)
    logit_scale = ls_state.params
    logit_scale = jax.tree_map(lambda a: a[0], logit_scale)
    loss, acc = ContrastiveLoss().apply(logit_scale, sim)
    return loss, acc

# ===== LOSS OVER BATCH OF TOKENS =====

# Takes:
#   passages: [BATCH, T, N_1], reviews: [BATCH, T, N_2] (tokenized)
# Returns:
#   grads for both encoders
def contrastive_grads(passages, reviews):
    b = passages.shape[0]
    indices = np.arange(b) # cross-ent labels
    indices = eo.rearrange(indices, "(cores examples) -> cores examples", cores = CORES)

    passages = eo.rearrange(passages, '(cores examples) t tokens -> cores examples t tokens', cores=CORES)
    reviews = eo.rearrange(reviews, '(cores examples) t tokens -> cores examples t tokens', cores=CORES)
    
    pass_encs = pmap(TextEncoder().apply)(pass_state.params, passages)
    pass_encs = eo.rearrange(pass_encs, "cores examples features -> (cores examples) features")
    pass_encs = util.l2norm(pass_encs)

    rev_encs = pmap(TextEncoder().apply)(rev_state.params, reviews)
    rev_encs = eo.rearrange(rev_encs, "cores examples features -> (cores examples) features")
    rev_encs = util.l2norm(rev_encs)

    def batch_stats(pass_encs, rev_encs):
        sim = np.einsum("i d, j d -> i j", pass_encs, rev_encs)
        logit_scale = ls_state.params
        logit_scale = jax.tree_map(lambda a: a[0], logit_scale)
        loss, acc = ContrastiveLoss().apply(logit_scale, sim, np.arange(BATCH_SIZE))
        return loss, acc

    batch_loss, batch_acc = batch_stats(pass_encs, rev_encs)
    
    def pass_loss(pass_params, logit_scale, sequences, labels):
        shard_pass_encs = TextEncoder().apply(pass_params, sequences)
        shard_pass_encs = util.l2norm(shard_pass_encs)
        sim = np.einsum("i d, j d -> i j", shard_pass_encs, rev_encs)
        loss, _ = ContrastiveLoss().apply(logit_scale, sim, labels)
        return loss

    def rev_loss(rev_params, logit_scale, sequences, labels):
        shard_rev_encs = TextEncoder().apply(rev_params, sequences)
        shard_rev_encs = util.l2norm(shard_rev_encs)
        sim = np.einsum("i d, j d -> i j", shard_rev_encs, pass_encs)
        loss, _ = ContrastiveLoss().apply(logit_scale, sim, labels)
        return loss

    # Input data split across TPUs
    @partial(pmap, axis_name='cores')
    def pass_grads(sequences, indices):
        # Grad calc across microbatches
        def microbatch(grad_accumulator, mcrobatch):
            seqs, inds = mcrobatch
            train_loss_fn = pass_loss_and_embeddings
            val_grad_fn = jax.value_and_grad(train_loss_fn)
            loss, grad = val_grad_fn(pass_state.params, ls_state.params, seqs, inds)
            grad_accumulator = jax.tree_multimap(lambda a, b: a + b, grad_accumulator, grad)
            return grad_accumulator
        
        sequences = eo.rearrange(sequences, "(microbatches examples) t tokens -> microbatches examples t tokens",
                                microbatches = MICROBATCHING_FACTOR)
        indices = eo.rearrange(indices, "(microbatches examples) -> microbatches examples",
                                microbatches = MICROBATCHING_FACTOR)

        grad = jax.lax.scan(microbatch,
                            jax.tree_map(lambda x: np.zeros_like(x), pass_state.params),
                            (sequences, indices))
        grad = jax.lax.pmean(grad, "cores")
        return grad

    @partial(pmap, axis_name='cores')
    def rev_grads(sequences, indices):
        # Grad calc across microbatches
        def microbatch(grad_accumulator, mcrobatch):
            seqs, inds = mcrobatch
            train_loss_fn = rev_loss_and_embeddings
            val_grad_fn = jax.value_and_grad(train_loss_fn)
            loss, grad = val_grad_fn(rev_state.params, ls_state.params, seqs, inds)
            grad_accumulator = jax.tree_multimap(lambda a, b: a + b, grad_accumulator, grad)
            return grad_accumulator
        
        sequences = eo.rearrange(sequences, "(microbatches examples) t tokens -> microbatches examples t tokens",
                                microbatches = MICROBATCHING_FACTOR)
        indices = eo.rearrange(indices, "(microbatches examples) -> microbatches examples",
                                microbatches = MICROBATCHING_FACTOR)

        grad = jax.lax.scan(microbatch,
                            jax.tree_map(lambda x: np.zeros_like(x), rev_state.params),
                            (sequences, indices))
        grad = jax.lax.pmean(grad, "cores")
        return grad

    p_grads = pass_grads(passages, indices)
    p_grads = jax.tree_map(lambda a: a[0], p_grads) 
    r_grads = rev_grads(reviews, indices)
    r_grads = jax.tree_map(lambda a: a[0], r_grads)

    return p_grads, r_grads, batch_loss, batch_acc

# ==== TOKENIZER AND TOKENIZATION ====
tokenizer = FlaxTokenizer()
def tok(string_batch):
    return tokenizer.tok(string_batch)

# Get tokenizer output over batch specified by inds
def get_batch_tokens(dataset, inds):
    batch = [dataset[ind] for ind in inds]
    pass_batch = [pair[0] for pair in batch]
    rev_batch = [pair[1] for pair in batch]

    pass_ids = tok(pass_batch) # -> [T x B x N_1]
    rev_ids = tok(rev_batch) # -> [T x B x N_2]

    return [pass_ids, rev_ids]

# Training Loop
pass_state = flax.jax_utils.replicate(pass_state)
rev_state = flax.jax_utils.replicate(rev_state)
ls_state = flax.jax_utils.replicate(ls_state)

dataset_size = len(dataset)
evalset_size = len(evalset)

skip_last_batch = dataset_size % BATCH_SIZE != 0
val_skip_last_batch = evalset_size % BATCH_SIZE != 0

total_steps = 0

for epoch in range(EPOCHS):
    batches_inds = util.generate_indices(dataset_size, BATCH_SIZE)
    if skip_last_batch: batches_inds = batches_inds[:-1]
    
    for batch_inds in batches_inds:
        passages, reviews = get_batch_tokens(dataset, batch_inds)
        
        passages = eo.rearrange(passages, 't b n -> b t n')
        reviews = eo.rearrange(reviews, 't b n -> b t n')

        p_grads, r_grads, batch_loss, batch_acc = contrastive_grads(passages, reviews)

        pass_state = pass_state.apply_gradients(grads = p_grads)
        rev_state = rev_state.apply_gradients(grads = r_grads)

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
            states = [unrp(pass_state).params, unrp(rev_state).params, unrp(ls_state).params]
            util.save_checkpoint(states)
            # Once every 10 saves, save copied backup
            if total_steps % (10 * CHECKPOINT_INTERVAL) == 0:
              util.save_checkpoint(states,
                              "/checkpoints", str(total_steps))
        
        total_steps += 1
