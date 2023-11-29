"""
Oddball tasks

TODO: summarize task
"""
# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

import sys
sys.path.append('../')
from train import train
from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from model.poly import PolyConfig
from task.oddball import FreeOddballTask

# <codecell>

task = FreeOddballTask(data_size=256)

# config = TransformerConfig(pure_linear_self_att=True)
# config = TransformerConfig(n_emb=None, n_out=6, n_layers=3, n_hid=128, use_mlp_layers=True, pure_linear_self_att=False)
# config = MlpConfig(n_out=6, n_layers=3, n_hidden=128)

config = PolyConfig(n_hidden=128, n_layers=1, n_out=6)
state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=50_000, lr=1e-4, l1_weight=1e-4)

# %%
xs, ys = next(task)

logits = state.apply_fn({'params': state.params}, xs)
print(logits.argmax(axis=1))
print(ys)

# %%
# TODO: solidify results <-- STOPPED HERE
# also plot transition from memorization to generalization in MNN and MLP

x = np.random.randn(*(1, 6, 2)) * 1
x[0, 4] = 4
plt.scatter(x[0,:,0], x[0,:,1], c=np.arange(6))
plt.colorbar()

logits = state.apply_fn({'params': state.params}, x)
logits

