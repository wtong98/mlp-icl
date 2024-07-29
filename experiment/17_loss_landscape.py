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
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, RfConfig
from task.function import PointTask, SameDifferent 

fig_dir = Path('fig/')

# <codecell>
### Markov process simulation
succ = 0
tot = 0

for _ in tqdm(range(10_000)):
    n_symbols = 100
    n_iters = 1000
    w1 = np.zeros(n_symbols)
    w2 = np.zeros(n_symbols)

    for _ in range(n_iters):
        if np.random.random() > 0.5:
            z1 = z2 = np.random.randint(n_symbols)
        else:
            z1, z2 = np.random.choice(n_symbols, size=2, replace=False)

        dec = w1[z1] + w2[z2]
        if dec > 0 or (dec == 0 and np.random.random() > 0.5):
            if z1 == z2:
                w1[z1] += 1
                w2[z2] += 1
            else:
                w1[z1] -= 1
                w2[z2] -= 1

    if np.sum(w1 > 0) > 0:
        succ += np.mean(w1 > 0)
        tot += 1

# print(w1)
# print(w2)
print(succ / tot)

# <codecell>
### Markov process simulation
n_symbols = 3
n_iters = 1000
w1 = np.zeros(n_symbols)
w2 = np.zeros(n_symbols)

for _ in range(n_iters):
    if np.random.random() > 0.5:
        z1 = z2 = np.random.randint(n_symbols)
    else:
        z1, z2 = np.random.choice(n_symbols, size=2, replace=False)

    dec = w1[z1] + w2[z2]
    if dec > 0 or (dec == 0 and np.random.random() > 0.5):
        if z1 == z2:
            w1[z1] += 1
            w2[z2] += 1
        else:
            w1[z1] -= 1
            w2[z2] -= 1

print(w1)
print(w2)

# <codecell>
### First-order simulation
n_points = 2
n_dims = 128
n_hidden = 512
alpha = 0.01  # learning strength

n_iters = 50

a0 = np.random.randn(n_hidden) / np.sqrt(n_hidden)
W0 = np.random.randn(n_hidden, 2*n_dims) / np.sqrt(2*n_dims)

a = np.copy(a0)
W = np.copy(W0)

sd_task = SameDifferent(n_dims=n_dims, n_symbols=n_points)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None, batch_size=1024)

for _, (x, y) in zip(range(n_iters), sd_task):
    x = x.reshape(x.shape[0], -1)

    dots = x @ W.T > 0
    signs = 2 * alpha * (y - 0.5)
    signed_x = x * np.expand_dims(signs, axis=1)
    xi = dots.T @ signed_x

    # a += np.diag(W0 @ xi.T)
    # W += np.expand_dims(a0, 1) * xi
    a += np.diag(W @ xi.T)
    W += np.expand_dims(a, 1) * xi

W = W.T
a = np.expand_dims(a, 1)

# <codecell>
x, y = next(test_task)
x = x.reshape(x.shape[0], -1)
out = jax.nn.relu(x @ W) @ a
preds = (out > 0).astype(int).flatten()

acc = np.mean(preds == y)
print('acc', acc)

pos_acc = np.mean(preds[y>0] == 1)
neg_acc = np.mean(preds[y==0] == 0)

print('pos_acc', pos_acc)
print('neg_acc', neg_acc)


# <codecell>
zs = sd_task.symbols
comp1 = zs @ W[:128,:] / alpha
comp2 = zs @ W[128:,:] / alpha

print(comp1[:,7])
print(comp2[:,7])

# <codecell>
succ = 0
n_iters = 10_000

for _ in range(n_iters):
    w1, w2, z1, z2 = np.random.randn(4, 512)
    w1 += z1
    w2 -= z1
    if w1 @ z1 + w2 @ z2 > 0:
        succ += 1

succ / n_iters

# %%
# W = W.T
# a = np.expand_dims(a, 1)
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


# <codecell>
n_points = 2
n_dims = 128
n_hidden = 512

sd_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True)
test_task = SameDifferent(n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True)

gamma0 = 1000
lr = gamma0 * 0.1
# lr = optax.exponential_decay(1, transition_steps=2000, decay_rate=2_000_000_000)
# lr = 1

config = MlpConfig(mup_scale=False,
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
                    gamma=None,
                    test_every=1000,
                    train_iters=5_000, 
                    lr=lr, optim=optax.sgd,
                    save_params=True,
                    seed=None)


# <codecell>
key = jax.random.PRNGKey(10)
sd_task = SameDifferent(n_dims=128, n_symbols=8, seed=5, reset_rng_for_data=False)
x, y = next(sd_task)

config1 = MlpConfig(mup_scale=False,
                    n_out=1, 
                    vocab_size=None, 
                    n_layers=1, 
                    n_hidden=512, 
                    feature_learning_strength=100,
                    use_bias=False,
                    act_fn='relu')

m1 = config1.to_model()
k, key = jax.random.split(key)
p1 = m1.init(k, x)['params']

def loss_fn(p):
    logits = m1.apply({'params': p}, x)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, labels=y)
    return loss.mean()

logits1 = m1.apply({'params': p1}, x)
loss1 = optax.losses.sigmoid_binary_cross_entropy(logits1, labels=y)
loss1.mean()

grads = jax.grad(loss_fn)(p1)
grads


config2= MlpConfig(mup_scale=False,
                    n_out=1, 
                    vocab_size=None, 
                    n_layers=1, 
                    n_hidden=512, 
                    feature_learning_strength=1,
                    use_bias=False,
                    act_fn='relu')

m2 = config2.to_model()
p2 = m2.init(k, x)['params']

def loss_fn2(p):
    logits = m2.apply({'params': p}, x)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, labels=y)

    gamma = 100

    logits = jax.lax.stop_gradient(logits)
    # num_pre = y - (1 - y) * jnp.exp((1 / gamma) * logits)
    # den_pre = y - (1 - y) * jnp.exp(logits)
    # factors = (1 + jnp.exp(logits)) / (1 + jnp.exp((1 / gamma) * logits))
    # scale_factors = (num_pre / den_pre) * factors

    log_fac = (1 - y) * logits * ((1 / gamma) - 1) \
                + jax.nn.softplus(logits) \
                - jax.nn.softplus((1 / gamma) * logits)
    loss = jnp.exp(log_fac + jnp.log(loss))
    # loss = jnp.exp(log_fac) * loss
    return loss.mean() / gamma

grads2 = jax.grad(loss_fn2)(p2)
grads2
# print(loss_fn2(p2))
# print(loss_fn(p1))

# def loss_fn(params):
#     logits = state.apply_fn({'params': params}, x)
#     train_loss = loss_func(logits, labels)

#     if loss == 'bce' and len(labels.shape) > 1:
#         assert logits.shape == train_loss.shape
#         train_loss = train_loss.mean(axis=-1)

#     assert len(train_loss.shape) == 1

#     l1_term = l1_weight * l1_loss(params)

#     if gamma is not None:
#         logits = jax.lax.stop_gradient(logits)
#         num_pre = labels - (1 - labels) * jnp.exp((1 / gamma) * logits)
#         den_pre = labels - (1 - labels) * jnp.exp(logits)
#         factors = (1 + jnp.exp(logits)) / (1 + jnp.exp((1 / gamma) * logits))
#         scale_factors = (num_pre / den_pre) * factors
#         train_loss = scale_factors * train_loss

#     return train_loss.mean() + l1_term





# <codecell>
x, _ = next(sd_task)
gamma=gamma0

logits = state.apply_fn({'params': state.params}, x)
avg_logit = jnp.mean(jnp.abs(logits))
scale_factor = (1 + jnp.exp(avg_logit)) / (1 + jnp.exp((1 / gamma) * avg_logit))

print(avg_logit)
scale_factor

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
pc1 = vh[0].reshape(256, 128)
# pc1 = w[12].reshape(256, 128)
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

# <codecell>
df = collate_dfs('remote/17_loss_landscape/generalize')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
    ], index=['name', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df \
        .groupby(['name', 'n_symbols', 'n_dims'], as_index=False) \
        .mean()

names = mdf['name'].unique()
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

for name, ax in zip(names, axs.ravel()):
    cdf = mdf[mdf['name'] == name].pivot(index='n_symbols', columns='n_dims', values='acc_seen')
    sns.heatmap(cdf, ax=ax, vmin=0.5, vmax=1)
    ax.set_title(name)

fig.suptitle('acc_seen')
fig.tight_layout()
plt.savefig(fig_dir / 'same_diff_acc_seen.png')

# <codecell>
# TODO: reconfigure to add error shading

n_dims = np.sort(plot_df['n_dims'].unique())
n_symbols = np.sort(plot_df['n_symbols'].unique())
threshold = 0.7

mdf = plot_df \
        .groupby(['name', 'n_symbols', 'n_dims'], as_index=False) \
        .mean()

# names = mdf['name'].unique()
names = ['MLP (RF)']
for name in names:
    if 'g' in name:
        continue

    cdf = mdf[mdf['name'] == name].pivot(index='n_symbols', columns='n_dims', values='acc_unseen')
    tdf = pd.DataFrame(np.argwhere(np.cumsum(cdf > threshold, axis=0) == 1), columns=['n_symbols', 'n_dims'])
    tdf['n_symbols'] = n_symbols[tdf['n_symbols']]
    tdf['n_dims'] = n_symbols[tdf['n_dims']]

    sns.lineplot(tdf, x='n_dims', y='n_symbols', marker='o', linestyle='dashed', label=name)

# xs = tdf['n_dims']
# plt.plot(xs, 0.3 * xs**2)

plt.gca().set_yscale('log')
plt.gca().set_xscale('log')

# plt.savefig(fig_dir / 'same_diff_data_diversity.png')

# <codecell>
df = collate_dfs('remote/17_loss_landscape/generalize_rf')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_symbols,
        row['train_task'].n_dims,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
    ], index=['name', 'n_symbols', 'n_dims', 'acc_seen', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)

plot_df['neg_log_acc_unseen'] = -np.log(plot_df['acc_unseen'])
plot_df

# %%
g = sns.lineplot(plot_df, x='n_dims', y='neg_log_acc_unseen', hue='n_symbols', marker='o')

ds = np.unique(plot_df['n_dims'].to_numpy())
ds = ds.astype('float')
plt.plot(ds, 0.0001 * ds**(2))
plt.axhline(y=-np.log(0.5))

g.set_yscale('log')
g.set_xscale('log')
