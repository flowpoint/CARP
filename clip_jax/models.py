import transformers
import einops as eo
import jax
import jax.numpy as np
import optax
import flax
import flax.linen as nn

from constants import *
from util import l2norm

from transformers.models.bert.modeling_flax_bert import FlaxBertForMaskedLMModule

special_token_dict = {'cls_token':'[CLS]', 'mask_token':'[quote]'}
model_name = "roberta-large"
tokenizer_name = "bert-base-uncased"

def get_model_config():
  tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
  config = transformers.RobertaConfig()
  tokenizer.add_special_tokens(special_token_dict)
  config.vocab_size = len(tokenizer)

  return config

class FlaxTokenizer():
  def __init__(self):
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    self.tokenizer.add_special_tokens(special_token_dict)
  
  def add_cls(self, string_batch):
    return [s + "[CLS]" for s in string_batch]
  
  # -> [B x T x N]
  # Where T is arbitrarily how many things the HF model being used needs from
  # tokenizer
  def tok(self, string_batch):
    res_dict = self.tokenizer(self.add_cls(string_batch),
                          return_tensors = 'jax',
                          padding = True,
                          truncation = True,
                          max_length = N_CTX)
    b, n = res_dict['input_ids'].shape
    position_ids = np.broadcast_to(np.arange(n)[None, :], (b, n))
    res = np.stack([res_dict['input_ids'],
                     res_dict['attention_mask'],
                     res_dict['token_type_ids'],
                     position_ids])
    return res

from transformers.models.bert.modeling_flax_bert import FlaxBertForMaskedLMModule
from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaForMaskedLMModule

class LMEmbedder(nn.Module):
  def setup(self):
    self.model = FlaxRobertaForMaskedLMModule(get_model_config())

  # Input assumed as tuple of everything the HF model needs
  def __call__(self, inp):
    mask = inp[1]
    out = self.model(*inp,
                        output_hidden_states = True)
    # [1] gets hidden states
    # [-2] gets last hidden state ([-1] is logits)
    hidden = out[1][-2]

    # Mask out pad tokens and sum
    sum_mask = eo.repeat(mask, 'b n -> b n d', d = hidden.shape[-1])
    embed = np.sum(hidden * sum_mask, axis = 1)

    return l2norm(embed)

class TextEncoder(nn.Module):
  # Input assumed as token mask pair
  # (token, mask) [2 x B x N]
  @nn.compact
  def __call__(self, x):
    x = LMEmbedder()(x)
    x = nn.Dense(LATENT_DIM)(x) # [B x D]
    return x # -> [B, D]

def cross_entropy(logits, labels, axis = -1):
    logprobs = jax.nn.log_softmax(logits, axis=axis)
    nll = np.take_along_axis(logprobs, np.expand_dims(labels, axis=axis), axis = axis)
    ce = -np.mean(nll)
    return ce

class ContrastiveLoss(nn.Module):
  @nn.compact
  def __call__(self, inp): # Assumed encodings
    # inp: [2 x B x D]
    x, y = inp

    x = l2norm(x)
    y = l2norm(y)

    scale = self.param('logit_scale', # aka temperature
                       lambda rng, shape: np.ones(shape) * np.log(1 / 0.07), [])
    
    logits = x @ y.T * np.exp(scale)
    labels = np.arange(x.shape[0])

    loss_x = cross_entropy(logits, labels, axis = 1)
    loss_y = cross_entropy(np.transpose(logits), labels, axis = 1)

    acc_x = np.sum((np.argmax(logits, axis = 1) == labels))
    acc_y = np.sum((np.argmax(logits, axis = 0) == labels))

    return (loss_x + loss_y) / 2, (acc_x + acc_y) / x.shape[0] / 2

# Same as above but divided to calculate losses seperately
# Doesn't do normalization
class ContrastiveLoss2(nn.Module):
    def setup(self):
        self.scale = self.param('logit_scale', 
                                lambda rng, shape: np.ones(shape) * np.log(1 / 0.07), [])

    def __call__(self, logits, labels):
        logits = logits * np.exp(self.scale)
        loss = cross_entropy(logits, labels, axis=1)
        acc = np.mean(np.argmax(logits, axis = 0) == labels)
        return loss, acc
        

# Loading pre trained HF checkpoints is kind of tricky
# Needs state of text encoder
def load_pretrained(state):
  if(type(state) != dict): # It's either frozen or not
    state = state.unfreeze()
 
  model_class = transformers.FlaxRobertaForMaskedLM
  cfg = get_model_config()
  pretrained_model = model_class(cfg).from_pretrained(model_name, config = cfg, ignore_mismatched_sizes = True)
  
  pretrained_state = pretrained_model.params # {'bert', 'cls'}

  state['params']['LMEmbedder_0']['model'] = pretrained_state

  return flax.core.frozen_dict.freeze(state)
