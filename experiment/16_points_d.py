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
from task.function import PointTask, SameDifferent, SameDifferentToken

fig_dir = Path('fig/final')

# <codecell>
xor_points = [
    ([-1, -1], 1),
    ([-1,  1], 0),
    ([ 1, -1], 0),
    ([ 1,  1], 1),
]

task = PointTask(xor_points, batch_size=128)

# <codecell>
points = [
    ([-1,  0], 1),
    ([ 1,  0], 1),
    ([-2,  0], 0),
    ([ 2,  0], 0),
]

task = PointTask(points, batch_size=128)

# <codecell>

n_out = 1

config = MlpConfig(n_out=n_out, n_layers=1, n_hidden=2048, act_fn='relu')
# config = RfConfig(n_in=2, n_out=n_out, n_hidden=512)

state, hist = train(config, data_iter=iter(task), loss='bce', test_every=1000, train_iters=10_000, lr=1e-4)
# %%
# x = np.array([[1, 1], [1, 0.6]])
xs = np.linspace(-4, 4, 100)
ys = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(xs, ys)

inp_x = np.stack((X.flatten(), Y.flatten()), axis=-1)
zs = state.apply_fn({'params': state.params}, inp_x)
Z = zs.reshape(X.shape)
plt.contourf(X, Y, Z>0)
plt.colorbar()

plt.contour(X, Y, Z)
plt.xlabel('$x$')
plt.ylabel("$x'$")

# plt.savefig('fig/xor.png')

# <codecell>
# NOTE: should fix bias to zero
import jax
params = jax.tree_map(np.array, state.params)

# a = params['readout']['kernel']
# key = jax.random.PRNGKey(config.seed)
# scale = config.scale / np.sqrt(config.n_hidden)
# W = scale * jax.random.normal(key, (config.n_in, config.n_hidden))


W = params['Dense_0']['kernel']
a = params['Dense_1']['kernel']


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
# plt.xlim((-0.1, 0.1))
# plt.ylim((-0.1, 0.1))
plt.tight_layout()
# plt.savefig('fig/xor_proj_rf.png')

# %%
n_points = 256
n_dims = 16

sd_task = SameDifferent(n_dims=n_dims, soft=False, radius=1)
task = Finite(sd_task, data_size=n_points)

config = MlpConfig(n_out=1, vocab_size=None, n_layers=1, n_hidden=512, act_fn='relu')

state, hist = train(config, data_iter=iter(task), test_iter=iter(sd_task), loss='bce', test_every=1000, train_iters=100_000, lr=1e-4)

# <codecell>
# task.sample_seen = False
task = sd_task
xs, ys = next(task)

pred = state.apply_fn({'params': state.params}, xs)
pred_ys = (pred > 0).astype(float)
print('PREDS', pred_ys)
print('ACC', np.mean(pred_ys == ys))


# <codecell>
n_points = 128
n_dims = 1024
seed = 5

task = SameDifferentToken(n_vocab=2*n_points, n_seen=n_points, seed=5)
test_task = SameDifferentToken(n_vocab=2*n_points, n_seen=n_points, seed=5, sample_seen=False)

config = MlpConfig(n_out=1, vocab_size=2*n_points, n_emb=n_dims, n_layers=1, n_hidden=512, act_fn='relu')
state, hist = train(config, data_iter=iter(task), test_iter=iter(test_task), loss='bce', test_every=1000, train_iters=10_000, lr=1e-4)

# %%
task.sample_seen = False
xs, ys = next(task)

pred = state.apply_fn({'params': state.params}, xs)
pred_ys = (pred > 0).astype(float)
print('PREDS', pred_ys)
print('ACC', np.mean(pred_ys == ys))

# <codecell>
import jax
params = jax.tree_map(np.array, state.params)

emb = params['Embed_0']['embedding']

emb_seen = emb[:n_points]

# u, s, vh = np.linalg.svd(emb_seen)
# plt.plot(s)
emb_seen = emb_seen / np.linalg.norm(emb_seen, axis=1, keepdims=True)
plt.imshow(emb_seen @ emb_seen.T > 0)
plt.colorbar()

# <codecell>
emb_unseen = emb[n_points:]
# emb_unseen = np.random.randn(*emb_unseen.shape) * 1 / np.sqrt((n_dims))
u, s, vh = np.linalg.svd(emb_unseen)
plt.plot(s)

# %%
plt.hist(params['Dense_0']['bias'])

# <codecell>
params['Dense_1']['bias']

# <codecell>
acc = [m.accuracy.item() for m in hist['test']]
plt.plot(acc)

# <codecell>
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
idxs = np.argsort(a.flatten())
W_sort = W_net[:,idxs]
dots = []

for idx in range(512):
    x, y = W_sort[:n_dims, [idx]], W_sort[n_dims:, [idx]]
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    dots.append((x.T @ y / (x_norm * y_norm)).item())

dots = np.array(dots)
dots = np.arccos(dots) * (360 / 2 / np.pi)
plt.plot(dots)

# <codecell>
u, s, vh = np.linalg.svd(W)
cs = np.cumsum(s**2) / np.sum(s**2)
plt.plot(cs, 'o--')
# plt.plot(s)


# <codecell>
for d, sign in zip(W_net.T, a):
    color='red'
    if sign > 0:
        color = 'blue'

    plt.arrow(0, 0, d[0], d[1], alpha=0.03, color=color)

plt.gca().set_aspect('equal')
plt.tight_layout()
# plt.savefig('fig/xor_proj_rf.png')
