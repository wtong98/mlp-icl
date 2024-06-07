"""
Probing a potential error in Boix paper about MLP's failing to learn relational
reasoning
"""

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
from model.mlp import MlpConfig, DotMlpConfig, RfConfig
from model.transformer import TransformerConfig
from task.function import SameDifferent, SameDifferentToken

fig_dir = Path('fig/final')
# set_theme()

# <codecell>
df = collate_dfs('remote/15_same_diff/generalize')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['test_task'].n_vocab,
        row['info']['acc_seen'].item(),
        row['info']['acc_unseen'].item(),
        row['config']['n_layers'],
        row['config']['n_emb'],
        row['config']['n_hidden'],
    ], index=['name', 'n_vocab', 'acc_seen', 'acc_unseen', 'n_layers', 'n_emb', 'n_hidden'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
fg = sns.relplot(plot_df, x='n_vocab', y='acc_seen', hue='name', row='n_emb', col='n_hidden', kind='line')

for ax in fg.axes.ravel():
    ax.set_xscale('log', base=2)

plt.savefig('same_diff_acc_seen.png')

# <codecell>
fg = sns.relplot(plot_df, x='n_vocab', y='acc_unseen', hue='name', row='n_emb', col='n_hidden', kind='line')

for ax in fg.axes.ravel():
    ax.set_xscale('log', base=2)

plt.savefig('same_diff_acc_unseen.png')

# <codecell>


g = sns.lineplot(plot_df, x='n_vocab', y='acc_unseen', hue='name', marker='o')
g.set_xscale('log', base=2)

g.axhline(y=0.5, linestyle='dashed', color='k', alpha=0.3)
# g.set_xlabel(r'$|\mathcal{X}|$')
g.set_xlabel('Vocabulary size')
g.set_ylabel('Accuracy on unseen tokens')

fig = g.figure
fig.tight_layout()
# fig.savefig(fig_dir / 'same_diff_mlp_acc.svg')


# <codecell>
# Quick-n-dirty plotting
n_out = 1

results = []

tasks = [
    SameDifferent(n_dims=2, seed=None, soft=True),
    SameDifferent(n_dims=2, seed=None, soft=False),
]

configs = [
    MlpConfig(n_out=n_out, n_layers=2, n_hidden=128, act_fn='relu'),
    TransformerConfig(pos_emb=True, n_out=n_out, n_layers=2, n_heads=2, n_hidden=128, n_mlp_layers=2, layer_norm=True, max_len=128)
]

for t in tasks:
    for c in configs:
        state, hist = train(c, data_iter=iter(t), loss='bce', test_every=1000, train_iters=5_000, lr=1e-4)
        xs, ys = next(t)
        ys_pred = state.apply_fn({'params': state.params}, xs)

        ys_pred = (ys_pred > 0).astype(float)
        acc = np.mean(ys == ys_pred)
        print('ACC', acc)

        results.append({
            'soft': t.soft,
            'model': str(type(c)),
            'acc': acc.item()
        })

# <codecell>
df = pd.DataFrame(results)
g = sns.barplot(df, x='soft', y='acc', hue='model')

plt.tight_layout()
plt.savefig('fig/same_different_raw.png')


# <codecell>
n_out = 1
n_vocab = 8000
n_seen = n_vocab // 2

task = SameDifferentToken(n_vocab=n_vocab, n_seen=n_seen, seed=5)
# sd_task = SameDifferent(n_dims=128)
# task = Finite(sd_task, data_size=n_seen)

# config = MlpConfig(n_out=n_out, vocab_size=None, n_layers=1, n_hidden=512, act_fn='relu')
config = MlpConfig(n_out=n_out, vocab_size=n_vocab, n_layers=3, n_emb=128, n_hidden=2, act_fn='relu')
# config = TransformerConfig(pos_emb=True, n_out=n_out, n_layers=2, n_heads=2, n_hidden=128, n_mlp_layers=2, layer_norm=True, max_len=128)

state, hist = train(config, data_iter=iter(task), loss='bce', test_every=1000, train_iters=10_000, lr=1e-4)

# <codecell>
task.sample_seen = False
# task = sd_task
xs, ys = next(task)

pred = state.apply_fn({'params': state.params}, xs)
pred_ys = (pred > 0).astype(float)
print('PREDS', pred_ys)
print('ACC', np.mean(pred_ys == ys))

