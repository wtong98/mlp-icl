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
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from train import train
from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from model.poly import PolyConfig
from task.oddball import FreeOddballTask


def free_oddball_experiment(config, train_iters=50_000, lr=1e-4, l1_weight=1e-4, **task_kwargs):
    task = FreeOddballTask(**task_kwargs)
    state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=train_iters, lr=lr, l1_weight=l1_weight)
    return state, hist


@dataclass
class Case:
    name: str
    config: dataclass
    experiment: Callable
    experiment_args: dict = field(default_factory=dict)
    state = None
    hist = None
    info: dict = field(default_factory=dict)

    def run(self):
        self.state, self.hist = self.experiment(self.config, **self.experiment_args)


def eval_cases(all_cases):
    eval_task = FreeOddballTask(batch_size=1024)
    for c in tqdm(all_cases):
        xs, ys = next(eval_task)
        logits = c.state.apply_fn({'params': c.state.params}, xs)
        preds = logits.argmax(axis=1)
        eval_acc = np.mean(ys == preds)
        print('ACC', eval_acc)

        c.info['eval_acc'] = eval_acc
# <codecell>
# In-distribution performance
n_iters = 3
n_out = 6

all_cases = []
for _ in range(n_iters):
    all_cases.extend([
        Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=128), free_oddball_experiment, experiment_args={}),
        Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=128, use_mlp_layers=True), free_oddball_experiment, experiment_args={}),
        Case('MNN', MlpConfig(n_out=n_out, n_layers=1, n_hidden=128), free_oddball_experiment, experiment_args={}),
    ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases)

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
plt.savefig('fig/oddball_in_dist_gen.png')

# <codecell>
# Memorization to generalization point
n_iters = 5
n_out = 6
data_sizes = [32, 64, 128, 256, 512, 1024]

all_cases = []
for _ in range(n_iters):
    for sz in data_sizes:
        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=128), free_oddball_experiment, experiment_args={
                'data_size': sz,'n_retry_if_missing_labels': 3, 'train_iters': 10_000
            }),
            # Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=128, use_mlp_layers=True), free_oddball_experiment, experiment_args={
            #     'data_size': sz,'n_retry_if_missing_labels': 3
            # }),
            Case('MNN', MlpConfig(n_out=n_out, n_layers=1, n_hidden=128), free_oddball_experiment, experiment_args={
                'data_size': sz,'n_retry_if_missing_labels': 3, 'train_iters': 10_000
            }),
        ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases)

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
task = FreeOddballTask(data_size=256)

# config = TransformerConfig(pure_linear_self_att=True)
# config = TransformerConfig(n_emb=None, n_out=6, n_layers=3, n_hid=128, use_mlp_layers=True, pure_linear_self_att=False)
# config = MlpConfig(n_out=6, n_layers=3, n_hidden=128)

config = PolyConfig(n_hidden=128, n_layers=1, n_out=6)
state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=50_000, lr=1e-4, l1_weight=1e-4)

# %%
xs, ys = next(task)

logits = state.apply_fn({'params': state.params}, xs)
print(logits.argmax(axis=1))
print(ys)

# %%
# also plot transition from memorization to generalization in MNN and MLP

x = np.random.randn(*(1, 6, 2)) * 1
x[0, 4] = 4
plt.scatter(x[0,:,0], x[0,:,1], c=np.arange(6))
plt.colorbar()

logits = state.apply_fn({'params': state.params}, x)
logits

