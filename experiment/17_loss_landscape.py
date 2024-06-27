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
n_points = 8
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

gamma0 = 10000
lr = gamma0 * 0.1

config = MlpConfig(mup_scale=True, 
                   n_out=1, 
                   vocab_size=None, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   feature_learning_strength=gamma0,
                   use_bias=False,
                   act_fn='relu')

state, hist = train(config, 
                    data_iter=iter(sd_task), 
                    test_iter=iter(test_task), 
                    loss='bce', 
                    test_every=1000,
                    train_iters=20_000, 
                    lr=lr, optim=optax.sgd,
                    save_params=True)


# <codecell>
a = [p['Dense_1']['kernel'].flatten() for p in hist['params']]

a = jnp.stack(a)
u, s, avh = jnp.linalg.svd(a)
# plt.plot(np.cumsum(s**2) / np.sum(s**2), '--o')

pcs = avh[:2,:]
pcs.shape

a_pc = pcs @ a.T
plt.scatter(a_pc[0], a_pc[1], c=np.arange(len(a)))
# plt.plot(a_pc[0], '--o')

# <codecell>
a_unit = a / np.linalg.norm(a, axis=1, keepdims=True)
au_pc = pcs @ a_unit.T
plt.scatter(au_pc[0], au_pc[1], c = np.arange(len(a)))


# <codecell>
w = [p['Dense_0']['kernel'].flatten() for p in hist['params']]
w = np.array(jnp.stack(w))

u, s, vh = np.linalg.svd(w)
plt.plot(np.cumsum(s**2) / np.sum(s**2), '--o')

# <codecell>
pcs = vh[:3,:]
pcs.shape

w_pc = pcs @ w.T
plt.scatter(w_pc[0], w_pc[1], c=np.arange(len(w)))
# plt.plot(a_pc[0], '--o')

# <codecell>
w_unit = w / np.linalg.norm(w, axis=1, keepdims=True)
wu_pc = pcs @ w_unit.T
plt.scatter(wu_pc[0], wu_pc[1], c=np.arange(len(w)))

# <codecell>
# pc1 = vh[0].reshape(256, 128)
pc1 = w[12].reshape(256, 128)
a = hist['params'][-1]['Dense_1']['kernel']

w1, w2 = pc1[:128, :], pc1[128:,:]
w1 = w1 / np.linalg.norm(w1, axis=0, keepdims=True)
w2 = w2 / np.linalg.norm(w2, axis=0, keepdims=True)
dots = np.diag(w1.T @ w2)

plt.scatter(avh[0], dots)

# <codecell>
params = jax.tree.map(np.array, state.params)
W = params['Dense_0']['kernel']
a = params['Dense_1']['kernel']

W_net = W * np.abs(a.T)
# W_net = W
W_net.shape

# <codecell>
# W_norms = np.linalg.norm(W_net, axis=0)
W_norms = np.linalg.norm(W, axis=0)
# plt.hist(W_norms)

signs = (a > 0).flatten().astype(int)
plt.scatter(a, W_norms)

# <codecell>
plt.hist(a)

# <codecell>
# idxs = np.argsort(W_norms)
plt.gcf().set_size_inches(4,3)
idxs = np.argsort(a.flatten())
W_sort = W[:,idxs]
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
