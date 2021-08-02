import jax
import jax.numpy as np
import flax

from constants import *
import util import device_split

def test_pmap(state):
    state = flax.jax_utils.replicate(state)
    batch = np.ones((16, 4, 8, 16))
    batch = device_split(batch)
    batch = np.stack(batch)

    def test_fn(state, batch):
        return 1

    test_fn = jax.pmap(test_fn)

    try:
        print(test_fn(state, batch))
    except Exception as e:
        print(e)

    state = flax.jax_utils.unreplicate(state)
