"""
Exploring the loss landscape, and why some optimizers succeed where others
fail.
"""

# <codecell>
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, RfConfig
from task.function import PointTask, SameDifferent 

# <codecell>
n_points = 16
n_dims = 128
n_hidden = 128

def mup_sgd(learning_rate):
    base_lr = learning_rate
    return optax.multi_transform({
        'in_weight': optax.sgd(base_lr * n_hidden),
        'out_weight': optax.sgd(base_lr * n_hidden),
        'biases': optax.sgd(base_lr / n_hidden),
    }, param_labels={
        'Dense_0': {'bias': 'biases', 'kernel': 'in_weight'},
        'Dense_1': {'bias': 'biases', 'kernel': 'out_weight'}
    })

sd_task = SameDifferent(n_dims=n_dims, n_symbols=n_points)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None)

config = MlpConfig(mup_scale=True, 
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   feature_learning_strength=100,
                   act_fn='relu')

state, hist = train(config, 
                    data_iter=iter(sd_task), 
                    test_iter=iter(test_task), 
                    loss='bce', 
                    test_every=1_000, 
                    train_iters=10_000, 
                    lr=10, optim=optax.sgd,
                    save_params=True)


# <codecell>
len(hist['params'])

# <codecell>
params = jax.tree.map(np.array, state.params)
W = params['Dense_0']['kernel']
a = params['Dense_1']['kernel']

W_net = W * np.abs(a.T)
# W_net = W
W_net.shape

# <codecell>
W_norms = np.linalg.norm(W_net, axis=0)
# plt.hist(W_norms)

signs = (a > 0).flatten().astype(int)
plt.scatter(a, W_norms)

# <codecell>
# idxs = np.argsort(W_norms)
plt.gcf().set_size_inches(4,3)
idxs = np.argsort(a.flatten())
W_sort = W_net[:,idxs]
dots = []

for idx in range(n_hidden):
    x, y = W_sort[:n_dims, [idx]], W_sort[n_dims:, [idx]]
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    dots.append((x.T @ y / (x_norm * y_norm)).item())

dots = np.array(dots)
print(dots)
# dots = np.arccos(dots) * (360 / 2 / np.pi)
plt.scatter(a.flatten()[idxs], dots)
plt.xlabel('$a_i$')
plt.ylabel('cosine distance')
plt.tight_layout()
# plt.savefig('fig/cosine_dists.png')
# %%
