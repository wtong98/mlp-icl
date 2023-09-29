"""
Symbolic Distance

This experiment measures the distribution of symbolic distance in MLPs trained
under different variants of the TI task. It appears that models exposed to
adjacencies >1 quickly exhibit a biologically-realistic symbolic distance
effect that is absence in the =1 case.
"""

# <codecell>
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.append('../')
from model import ModelConfig, train
from task.ti import TiTask


# <codecell>
vocab_size = 5
task = TiTask(dist=[1,2,3,4])

config = ModelConfig(vocab_size=vocab_size, train_iters=1_000, test_every=100)
state, hist = train(config, data_iter=iter(task))

# <codecell>
logits = [state.apply_fn({'params': state.params}, jnp.array([[0, i]]).astype(jnp.int32)) for i in range(1, vocab_size)]

# %%
plt.plot(logits)