"""Experimentation with XOR and SameDifferent generalization"""
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

fig_dir = Path('fig/final')


# <codecell>
df = collate_dfs('remote/16_points_d/generalize')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['params']['n_vocab'],
        row['info']['params']['n_dims'],
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
    ], index=['name', 'n_vocab', 'n_dims', 'acc_seen', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df \
        .groupby(['name', 'n_vocab', 'n_dims'], as_index=False) \
        .mean()

names = mdf['name'].unique()
fig, axs = plt.subplots(3, 1, figsize=(6, 12))

for name, ax in zip(names, axs.ravel()):
    cdf = mdf[mdf['name'] == name].pivot(index='n_vocab', columns='n_dims', values='acc_seen')
    sns.heatmap(cdf, ax=ax, vmin=0.5, vmax=1)
    ax.set_title(name)

fig.suptitle('acc_seen')
fig.tight_layout()

# <codecell>
# TODO: reconfigure to add error shading

n_dims = plot_df['n_dims'].unique()
n_vocab = plot_df['n_vocab'].unique()
threshold = 0.9

mdf = plot_df \
        .groupby(['name', 'n_vocab', 'n_dims'], as_index=False) \
        .mean()

names = mdf['name'].unique()
for name in names:
    cdf = mdf[mdf['name'] == name].pivot(index='n_vocab', columns='n_dims', values='acc_unseen')
    tdf = pd.DataFrame(np.argwhere(np.cumsum(cdf > threshold, axis=0) == 1), columns=['n_vocab', 'n_dims'])
    tdf['n_vocab'] = n_vocab[tdf['n_vocab']]
    tdf['n_dims'] = n_vocab[tdf['n_dims']]

    sns.lineplot(tdf, x='n_dims', y='n_vocab', marker='o', linestyle='dashed', label=name)



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
n_points = 16
n_dims = 128
n_hidden = 2048

def mup_sgd(learning_rate):
    base_lr = learning_rate
    return optax.multi_transform({
        'in_weight': optax.sgd(base_lr * n_hidden),
        'out_weight': optax.sgd(base_lr / n_hidden),
        'biases': optax.sgd(base_lr * n_hidden),
    }, param_labels={
        'Dense_0': {'bias': 'biases', 'kernel': 'in_weight'},
        'Dense_1': {'bias': 'biases', 'kernel': 'out_weight'}
    })

sd_task = SameDifferent(n_dims=n_dims, n_symbols=n_points)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None)

config = MlpConfig(mup_scale=True, n_out=1, vocab_size=None, n_layers=1, n_hidden=n_hidden, act_fn='relu')

state, hist = train(config, data_iter=iter(sd_task), test_iter=iter(test_task), loss='bce', test_every=1000, train_iters=50_000, lr=1e-3, optim=mup_sgd)
# state, hist = train(config, data_iter=iter(task), test_iter=iter(task), loss='bce', test_every=1000, train_iters=10_000, lr=1e-4)

# <codecell>
# task.sample_seen = False
xs, ys = next(test_task)

pred = state.apply_fn({'params': state.params}, xs)
pred_ys = (pred > 0).astype(float)
print('PREDS', pred_ys)
print('ACC', np.mean(pred_ys == ys))


# <codecell>
n_points = 16
n_dims = 1024
n_hidden = 512
seed = np.random.randint(999)

task = SameDifferent(task='token', n_symbols=2*n_points, n_seen=n_points, seed=seed)
test_task = SameDifferent(task='token', n_symbols=2*n_points, n_seen=n_points, seed=seed, sample_seen=False)

config = MlpConfig(n_out=1, vocab_size=2*n_points, n_emb=n_dims, n_layers=1, n_hidden=n_hidden, act_fn='relu')
state, hist = train(config, data_iter=iter(task), test_iter=iter(test_task), loss='bce', test_every=1000, train_iters=10_000, lr=1e-4, optim=optax.adam)

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
# plt.savefig('fig/non_monotonic_learned_emb.png')

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
params = jax.tree.map(np.array, state.params)
W = params['Dense_0']['kernel']
a = params['Dense_1']['kernel']

print(np.min(a[a<0]))

# NOTE: unclear what best scaling is (should be computable)
# a[a<0] = 0.036 * a[a<0]
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
### ROUGH EXPERIMENTS WITH MARKOV MODEL
def is_match(w, w_sens, z1_idx, z2_idx):
    if w[z1_idx] + w[z2_idx] > 0:
        return True
    
    elif w[z1_idx] + w[z2_idx] == 0:
        if z1_idx in w_sens or z2_idx in w_sens:
            return True
    
    return False

n_iters = 500
n_vocab = 2048
n_width = 128

rng = np.random.default_rng(None)

zs = np.zeros((n_vocab, 2*n_width))
ws = np.zeros((n_width, 2*n_vocab))
# ws_sens = rng.integers(0, n_vocab, size=(n_width, n_vocab // 2))
ws_sens = [rng.choice(n_vocab, size=n_vocab//2, replace=False) for _ in range(n_width)]
ws_sens = np.stack(ws_sens)

aa = rng.standard_normal(n_width) / np.sqrt(n_width)
aa = (aa > 0).astype(int)

# ws_pos = np.zeros(n_width // 2, n_vocab)
# ws_neg = np.zeros(n_width // 2, n_vocab)

task = SameDifferentToken(n_vocab=n_vocab, n_seen=n_vocab, seed=None, reset_rng_for_data=False)

for _ in tqdm(range(n_iters)):
    xs, ys = next(task)
    for x, y in zip(xs, ys):
    # x, y = next(zip(xs, ys))
        for i, (w, w_sens, a) in enumerate(zip(ws, ws_sens, aa)):
        # i, (w, w_sens, a) = list(enumerate(zip(ws, ws_sens, aa)))[44]
            if is_match(w, w_sens, x[0], x[1]):
                if a == 1:
                    upd = 1
                else:
                    upd = 1

                if a == y:
                    zs[x[0],i] += upd
                    zs[x[1],i+n_width] += upd
                    w[x[0]] += upd
                    w[x[1]+n_vocab] += upd
                else:
                    zs[x[0],i] -= upd
                    zs[x[1],i+n_width] -= upd
                    w[x[0]] -= upd
                    w[x[1]+n_vocab] -= upd



# <codecell>
w1, w2 = ws[:,:n_vocab], ws[:,n_vocab:]
# w1 = w1[aa==0]
# w2 = w2[aa==0]

w1_norm = np.linalg.norm(w1, axis=1, keepdims=True)
w2_norm = np.linalg.norm(w2, axis=1, keepdims=True)


diag = np.diag((w1 / w1_norm) @ (w2 / w2_norm).T)
plt.hist(diag, bins=100)

# <codecell>
np.sum(ws>0, axis=1)
aa

res = 0
for i in range(128):
    res += is_match(ws[1], ws_sens[1], i, i)

res / 128

mw = ws[1] - 16
mw[:128] @ mw[128:]


