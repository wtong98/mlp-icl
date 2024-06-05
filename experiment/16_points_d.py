# <codecell>
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, RfConfig
from task.function import PointTask

fig_dir = Path('fig/final')

# <codecell>
xor_points = [
    ([-1, -1], 1),
    ([-1,  1], 0),
    ([ 1, -1], 0),
    ([ 1,  1], 1),
]

task = PointTask(xor_points, batch_size=128)

n_out = 1

config = MlpConfig(n_out=n_out, n_layers=1, n_hidden=2048, act_fn='relu')
config = RfConfig(n_in=2, n_out=n_out, n_hidden=512)

state, hist = train(config, data_iter=iter(task), loss='bce', test_every=1000, train_iters=10_000, lr=1e-4)
# %%
# x = np.array([[1, 1], [1, 0.6]])
xs = np.linspace(-2, 2, 100)
ys = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(xs, ys)

inp_x = np.stack((X.flatten(), Y.flatten()), axis=-1)
zs = state.apply_fn({'params': state.params}, inp_x)
Z = zs.reshape(X.shape)
plt.contourf(X, Y, Z>0)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel("$x'$")

# plt.savefig('fig/xor.png')

# <codecell>
# NOTE: should fix bias to zero
import jax
params = jax.tree_map(np.array, state.params)

a = params['readout']['kernel']
key = jax.random.PRNGKey(config.seed)
scale = config.scale / np.sqrt(config.n_hidden)
W = scale * jax.random.normal(key, (config.n_in, config.n_hidden))


# W = params['Dense_0']['kernel']
# a = params['Dense_1']['kernel']


W_net = W * np.abs(a.T)
# W_net = W * a.T
W_net.shape

u, s, vh = np.linalg.svd(W_net)
u

# <codecell>

for d, sign in zip(W_net.T, a):
    color='red'
    if sign > 0:
        color = 'blue'

    plt.arrow(0, 0, d[0], d[1], alpha=0.03, color=color)

# plt.arrow(0, 0, 0.1, 0.1, color='red', alpha=0.7, linestyle='dashed')
# plt.arrow(0, 0, 0.1, -0.1, color='red', alpha=0.7, linestyle='dashed')
# plt.arrow(0, 0, -0.1, 0.1, color='red', alpha=0.7, linestyle='dashed')
# plt.arrow(0, 0, -0.1, -0.1, color='red', alpha=0.7, linestyle='dashed')

plt.gca().set_aspect('equal')
plt.xlim((-0.1, 0.1))
plt.ylim((-0.1, 0.1))
plt.tight_layout()
# plt.savefig('fig/xor_proj_rf.png')

# %%