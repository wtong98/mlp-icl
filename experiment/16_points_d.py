# <codecell>
from pathlib import Path

import jax
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
n_points = 16_000
n_dims = 64
n_hidden = 256

sd_task = SameDifferent(n_dims=n_dims, soft=False, radius=1)
task = Finite(sd_task, data_size=n_points)

config = MlpConfig(n_out=1, vocab_size=None, n_layers=1, n_hidden=n_hidden, act_fn='relu')

state, hist = train(config, data_iter=iter(task), test_iter=iter(sd_task), loss='bce', test_every=1000, train_iters=10_000, lr=1e-4)

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
n_hidden = 512
seed = np.random.randint(999)

task = SameDifferentToken(n_vocab=2*n_points, n_seen=n_points, seed=5)
test_task = SameDifferentToken(n_vocab=2*n_points, n_seen=n_points, seed=5, sample_seen=False)

config = MlpConfig(n_out=1, vocab_size=2*n_points, n_emb=n_dims, n_layers=1, n_hidden=n_hidden, act_fn='relu')
state, hist = train(config, data_iter=iter(task), test_iter=iter(test_task), loss='bce', test_every=1_000, train_iters=250_000, lr=1e-4)

# %%
task.sample_seen = False
xs, ys = next(task)

pred = state.apply_fn({'params': state.params}, xs)
pred_ys = (pred > 0).astype(float)
print('PREDS', pred_ys)
print('ACC', np.mean(pred_ys == ys))

# <codecell>
params = jax.tree_map(np.array, state.params)

emb = params['Embed_0']['embedding']

emb_seen = emb[:n_points]
# emb_seen = np.random.randn(*emb_seen.shape)
# emb_seen = emb

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
acc = [1 - m.accuracy.item() for m in hist['test']]
plt.plot(acc)
plt.xlabel('Training iterations (x1000)')
plt.ylabel('1 - accuracy')
plt.savefig('fig/non_monotonic_learned_emb.png')

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

# <codecell>
params = jax.tree_map(np.array, state.params)
W = params['Dense_0']['kernel']
a = params['Dense_1']['kernel']

print(np.min(a[a<0]))

# NOTE: unclear what best scaling is (should be computable)
a[a<0] = 0.036 * a[a<0]
# a[a<0] = -0.01

all_res = []

for _ in range(1000):
    z = np.random.randn(n_dims, 1) / np.sqrt(n_dims)
    x_pos = np.concatenate((z, z))
    x_neg = np.random.randn(2*n_dims, 1) / np.sqrt(n_dims)

    # W_pos = W[:,a.flatten()>0]
    # W_neg = W[:,a.flatten()<=0]
    # res = np.sum(x.T @ W_pos) > np.sum(x.T @ W_neg)

    res_pos = np.clip(x_pos.T @ W, 0, np.inf)
    res_pos = res_pos @ a > 0

    res_neg = np.clip(x_neg.T @ W, 0, np.inf)
    res_neg = res_neg @ a < 0

    all_res.append((res_pos, res_neg))

res_pos, res_neg = zip(*all_res)
print('POS', np.mean(res_pos))
print('NEG', np.mean(res_neg))

# <codecell>


# <codecell>
z = np.random.randn(1024, 1) / np.sqrt(1024)
# x = np.concatenate((z, z))
x = np.random.randn(2048, 1) / np.sqrt(1024)

res = np.clip(x.T @ W, 0, np.inf)
res @ a

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

# <codecell>
ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16_000, 32_000])
d = 1024

res = []
for n in ns:
    zs = np.random.randn(n, d) * np.sqrt(1/d)
    z1 = zs[zs[:,0] > 0]
    z2 = zs[zs[:,0] <= 0]

    z1_mean = np.sum(z1, axis=0, keepdims=True)
    z2_mean = np.sum(z2, axis=0, keepdims=True)

    z1_norm = np.linalg.norm(z1_mean)
    z2_norm = np.linalg.norm(z2_mean)
    # z1_norm = 1
    # z2_norm = 1

    ans = z1_mean @ z2_mean.T / (z1_norm * z2_norm)
    res.append(ans.item())

res
# <codecell>
# scale = 0.8 / np.sqrt(d)
scale = 0.3 / d   # up to the correct prefactor O(1/d)
power = 2

plt.plot(res, '--o')
plt.plot(-(ns**2 * scale) / (ns + ns**2 * scale), '--o')



# <codecell>
n = 800
d = 128

zs = np.random.randn(n, d) * np.sqrt(1/d)
z1 = zs[zs[:,0] > 0]
z2 = zs[zs[:,0] <= 0]

z1_mean = np.sum(z1, axis=0, keepdims=True)
z2_mean = np.sum(z2, axis=0, keepdims=True)

z1_norm = np.linalg.norm(z1_mean)
z2_norm = np.linalg.norm(z2_mean)
# z1_norm = 1
# z2_norm = 1

ans = z1_mean @ z2_mean.T / (z1_norm * z2_norm)
ans

# <codecell>
c = 0.64
d = 128
p = 0.5

1 / ((c / d) * (1 - p))