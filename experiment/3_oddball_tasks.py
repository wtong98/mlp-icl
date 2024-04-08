"""
Oddball tasks

The oddball task is an odd-one-out task inspired by Sable-Meyer's geometric
odd-one-out perceptual task. It appears that an MLP is able to learn this one
easily (and in-context?!), while a transformer performs poorly. This is a
fascinating outcome, but ill-suited towards our purpose of comparing MNNs to
transformers.
"""

# <codecell>
from dataclasses import dataclass, field
import functools
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from common import *

import sys
sys.path.append('../')
from train import train
from model.mlp import MlpConfig, DotMlpConfig
from model.transformer import TransformerConfig
from model.poly import PolyConfig
from task.oddball import FreeOddballTask, LineOddballTask


# <codecell>
# In-distribution performance
n_iters = 3
n_out = 6

common_args = {
    'loss': 'bce',
    'one_hot': True
}

all_cases = []
for _ in range(n_iters):
    all_cases.extend([
        Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=128), oddball_experiment, experiment_args=common_args),
        Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=128, use_mlp_layers=True, pos_emb=True), oddball_experiment, experiment_args=common_args),
        Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=128), oddball_experiment, experiment_args=common_args),
    ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases, eval_task=FreeOddballTask(n_choices=n_out))

# <codecell>
df = pd.DataFrame(all_cases)

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['eval_acc'].item(),
    ], index=['name', 'eval_acc'])

plot_df = df.apply(extract_plot_vals, axis=1)
# <codecell>
sns.barplot(plot_df, x='name', y='eval_acc')
plt.tight_layout()
plt.savefig('fig/oddball_in_dist_gen_sigmoid.png')

# <codecell>
# Memorization to generalization point
n_iters = 5
n_out = 6
data_sizes = [32, 64, 128, 256, 512, 1024]

all_cases = []
for _ in range(n_iters):
    for sz in data_sizes:
        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=128), oddball_experiment, experiment_args={
                'data_size': sz,'n_retry_if_missing_labels': 3, 'train_iters': 10_000
            }),
            # Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=128, use_mlp_layers=True), free_oddball_experiment, experiment_args={
            #     'data_size': sz,'n_retry_if_missing_labels': 3
            # }),
            Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=128), oddball_experiment, experiment_args={
                'data_size': sz,'n_retry_if_missing_labels': 3, 'train_iters': 10_000
            }),
        ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases, eval_task=FreeOddballTask())

# <codecell>
df = pd.DataFrame(all_cases)

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['experiment_args']['data_size'],
        row['info']['eval_acc'].item(),
    ], index=['name', 'data_size', 'eval_acc'])

plot_df = df.apply(extract_plot_vals, axis=1)
plot_df
# <codecell>
sns.barplot(plot_df, x='data_size', y='eval_acc', hue='name')
plt.tight_layout()
plt.savefig('fig/oddball_data_size_gen.png')

# <codecell>
task = FreeOddballTask(data_size=None)

# config = TransformerConfig(pure_linear_self_att=True)
# config = TransformerConfig(n_emb=None, n_out=6, n_layers=3, n_hid=128, use_mlp_layers=True, pure_linear_self_att=False)
config = MlpConfig(n_out=6, n_layers=3, n_hidden=128)

# config = PolyConfig(n_hidden=128, n_layers=1, n_out=6)
state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=10_000, lr=1e-4, l1_weight=1e-4)

# %%
xs, ys = next(task)

logits = state.apply_fn({'params': state.params}, xs)
print(logits.argmax(axis=1))
print(ys)

# %%
# also plot transition from memorization to generalization in MNN and MLP
# dists = 10**np.linspace(0.5, 2, num=100)
dists = np.linspace(0, 50, num=50)
all_targets = []
for d in tqdm(dists):
    x = np.random.randn(*(1, 6, 2)) * 0.5 + 5
    x[0, 4] = d
    # plt.scatter(x[0,:,0], x[0,:,1], c=np.arange(6))
    # plt.colorbar()

    logits = state.apply_fn({'params': state.params}, x)
    target = logits[0, 4]
    all_targets.append(target.item())


# %%
plt.plot(dists, all_targets, 'o--')
plt.xlabel('Distance')
plt.ylabel('Logit')
plt.savefig('fig/mlp_oddball_logit_dist.png')
# plt.plot(dists, dists)

"""
Some freaky stuff:
- decreasing spread of points sharpens line
- moving center will create a linear dip in the plot exactly
aligned with the center

- conclusion: it appears the model is computing the distance
to the center of the cluster exactly!
"""

# <codecell>
# EXPERIMENT WITH LINE ODDBALL TASK
n_iters = 3
n_out = 6
train_iters = 20_000

task_args = {
    'linear_dist': 10
}


all_cases = []
for _ in range(n_iters):
    all_cases.extend([
        Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=256), LineOddballTask(**task_args), train_args={'train_iters': train_iters}),
        Case('MLP (dot product)', MlpConfig(n_out=n_out, n_layers=3, n_hidden=256), LineOddballTask(**task_args), train_args={'train_iters': train_iters}),
        Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=256, n_mlp_layers=2, pos_emb=True), LineOddballTask(**task_args), train_args={'train_iters': train_iters}),
        Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=256), LineOddballTask(**task_args), train_args={'train_iters': train_iters})
        # TODO: redo long run properly with callbacks
    ])

# <codecell>
for case in tqdm(all_cases):
    print('CASE', case)
    case.run()

# <codecell>
dists = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for d in dists:
    key_name = f'eval_acc_dist_{d}'

    # will fail for dot-product model
    eval_task = LineOddballTask(linear_dist=d, batch_size=1024)
    eval_cases(all_cases, key_name=key_name, eval_task=eval_task, ignore_err=True)

    # will fail for non-dot-product model
    eval_task = LineOddballTask(linear_dist=d, batch_size=1024, with_dot_product_feats=True)
    eval_cases(all_cases, key_name=key_name, eval_task=eval_task, ignore_err=True)
        

# <codecell>
df = pd.DataFrame(all_cases)

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info'],
    ], index=['name', 'info'])

plot_df = df.apply(extract_plot_vals, axis=1)
eval_df = pd.DataFrame(plot_df['info'].to_list()) \
            .rename(columns=lambda name: int(name.split('_')[-1]))
plot_df = plot_df.join(eval_df) \
                 .drop('info', axis='columns') \
                 .melt(id_vars='name', var_name='distance', value_name='acc')

plot_df['acc'] = plot_df['acc'].apply(lambda x: x.item())

# <codecell>
plt.gcf().set_size_inches(12, 3)
ax = sns.barplot(plot_df, x='distance', y='acc', hue='name')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig('fig/line_oddball_gen_longer.png')


# <codecell>
n_choices = 6
# task = LineOddballTask(n_choices=n_choices, perp_dist=2, linear_dist=1, with_dot_product_feats=False)
task = FreeOddballTask(n_choices=n_choices, discrim_dist=5)

config = MlpConfig(n_out=n_choices, n_layers=3, n_hidden=256)
# config = PolyConfig(n_hidden=256, n_layers=1, n_out=n_choices)
# config = TransformerConfig(pos_emb=True, n_emb=None, n_out=n_choices, n_layers=3, n_hidden=128, n_mlp_layers=2, pure_linear_self_att=False)
# config = DotMlpConfig(n_out=n_choices, use_initial_proj=False, last_token_only=False, center_inputs=True)

state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=50_000, lr=1e-4, l1_weight=1e-4)

# %%
task = LineOddballTask(n_choices=n_choices, linear_dist=50, perp_dist=5, with_dot_product_feats=True)
xs, ys = next(task)

logits = state.apply_fn({'params': state.params}, xs)
print(logits.argmax(axis=1))
print(ys)

np.mean(logits.argmax(axis=1) == ys)
