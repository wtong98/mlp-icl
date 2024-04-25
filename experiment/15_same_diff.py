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
from model.mlp import MlpConfig, DotMlpConfig
from model.transformer import TransformerConfig
from task.function import SameDifferent 

fig_dir = Path('fig/final')

# <codecell>
# TODO: set up official plotting experiments

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
task = SameDifferent(n_dims=2, seed=5, soft=False)

# config = MlpConfig(n_out=n_out, n_layers=2, n_hidden=128, act_fn='relu')
config = TransformerConfig(pos_emb=True, n_out=n_out, n_layers=2, n_heads=2, n_hidden=128, n_mlp_layers=2, layer_norm=True, max_len=128)

state, hist = train(config, data_iter=iter(task), loss='bce', test_every=1000, train_iters=10_000, lr=1e-4)
# %%
x = np.array([[[1, 1], [1, 0.9]]])
state.apply_fn({'params': state.params}, x)
