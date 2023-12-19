"""
Experimenting with the match tasks, which is comparable to the
delayed match-to-sample task of the Griffiths papeer.

Observations:

- MLP learns this in-context
- Increasing radius is no problem --> performs some sort of dot product operation
- All models struggle to generalize from fixed --> scramble task
    - MNN seems to perform marginally better
    --> uses ring structure, not purely dot-product-driven

Tentative conclusion: with perfect data, in-context learning is well
within capacity of MLP (among other models)
"""

# <codecell>
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from train import train
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.match import RingMatch

from tqdm import tqdm


def match_experiment(config, task_class=RingMatch, train_iters=50_000, loss='ce', lr=1e-4, l1_weight=1e-4, **task_kwargs):
    task = task_class(**task_kwargs)
    state, hist = train(config, data_iter=iter(task), loss=loss, test_every=1000, train_iters=train_iters, lr=lr, l1_weight=l1_weight)
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


def eval_cases(all_cases, key_name='eval_acc', eval_task=None, ignore_err=False):
    if eval_task is None:
        eval_task = RingMatch(batch_size=1024)

    for c in tqdm(all_cases):
        try:
            xs, ys = next(eval_task)
            logits = c.state.apply_fn({'params': c.state.params}, xs)
            preds = logits.argmax(axis=1)
            eval_acc = np.mean(ys == preds)
            print('ACC', eval_acc)

            c.info[key_name] = eval_acc

        except Exception as e:
            if ignore_err:
                continue
            else:
                raise e


# <codecell>
# W/W_OUT SCRAMBLE GENERALIZATION
n_iters = 3
n_out = 6

common_args = {'train_iters': 20_000}
scramble_args = dict(scramble=True, **common_args)

all_cases = []
for _ in range(n_iters):
    all_cases.extend([
        Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=256), match_experiment, experiment_args=common_args),
        Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=256, use_mlp_layers=True, pos_emb=True), match_experiment, experiment_args=common_args),
        Case('MNN', MlpConfig(n_out=n_out, n_layers=1, n_hidden=256), match_experiment, experiment_args=common_args),

        Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=256), match_experiment, experiment_args=scramble_args),
        Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=256, use_mlp_layers=True, pos_emb=True), match_experiment, experiment_args=scramble_args),
        Case('MNN', MlpConfig(n_out=n_out, n_layers=1, n_hidden=256), match_experiment, experiment_args=scramble_args),
    ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases, 'in_dist_eval', eval_task=RingMatch(batch_size=1024))
eval_cases(all_cases, 'radius_eval', eval_task=RingMatch(radius=2, batch_size=1024))
eval_cases(all_cases, 'scramble_eval', eval_task=RingMatch(scramble=True, batch_size=1024))

# <codecell>
df = pd.DataFrame(all_cases)
df_info = pd.DataFrame(df['info'].tolist())
df_exp = pd.DataFrame(df['experiment_args'].tolist())
df_exp['scramble'][df_exp['scramble'].isna()] = False

df = df[['name']].join(df_exp[['scramble']]).join(df_info)
df = df.melt(id_vars=['name', 'scramble'], value_name='acc')
df['acc'] = df['acc'].astype('float')
df

# <codecell>
g = sns.FacetGrid(df, col='variable')
g.map_dataframe(sns.barplot, x='scramble', y='acc', hue='name')
g.add_legend()

# plt.tight_layout()
plt.savefig('fig/match_gen.png')

# <codecell>
# WIDTH PERFORMANCE
n_iters = 3
n_out = 6
widths = [32, 64, 128, 256, 512, 1024]

common_args = {'train_iters': 20_000}

all_cases = []
for _ in range(n_iters):
    for w in widths:
        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=w), match_experiment, experiment_args=common_args),
            Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=w, use_mlp_layers=True, pos_emb=True), match_experiment, experiment_args=common_args),
            Case('MNN', MlpConfig(n_out=n_out, n_layers=1, n_hidden=w), match_experiment, experiment_args=common_args),
        ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases, 'scramble_eval', eval_task=RingMatch(scramble=True, batch_size=1024))

# <codecell>
df = pd.DataFrame(all_cases)

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['config']['n_hidden'],
        row['info']['scramble_eval'].item(),
    ], index=['name', 'width', 'acc'])

df = df.apply(extract_plot_vals, axis=1)

# <codecell>
plt.gcf().set_size_inches(8, 3)
sns.barplot(df, x='width', y='acc', hue='name')
plt.tight_layout()
plt.savefig('fig/match_width_scramble.png')

# <codecell>

n_choices = 6
task = RingMatch(n_points=n_choices, scramble=False)

# config = MlpConfig(n_out=n_choices, n_layers=3, n_hidden=256)
# config = PolyConfig(n_out=n_choices, n_layers=1, n_hidden=512)

config = TransformerConfig(n_out=n_choices, n_hidden=256, n_layers=1, use_mlp_layers=False, pos_emb=True)

state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=50_000, lr=1e-4, l1_weight=1e-4)

# %%
task_scramble = RingMatch(n_points=n_choices, scramble=True, batch_size=1024)
task_large = RingMatch(n_points=n_choices, radius=1, batch_size=1024)

xs, labs = next(task_scramble)

logits = state.apply_fn({'params': state.params}, xs)
preds = logits.argmax(axis=1)
eval_acc = np.mean(labs == preds)
eval_acc
