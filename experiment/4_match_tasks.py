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
import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from common import *

import sys
sys.path.append('../')
from train import train
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.match import RingMatch, LabelRingMatch

match_experiment = functools.partial(experiment, task_class=RingMatch)

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
        Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=256), match_experiment, experiment_args=common_args),

        Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=256), match_experiment, experiment_args=scramble_args),
        Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=256, use_mlp_layers=True, pos_emb=True), match_experiment, experiment_args=scramble_args),
        Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=256), match_experiment, experiment_args=scramble_args),
    ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases, key_name='in_dist_eval', eval_task=RingMatch(batch_size=1024))
eval_cases(all_cases, key_name='radius_eval', eval_task=RingMatch(radius=2, batch_size=1024))
eval_cases(all_cases, key_name='scramble_eval', eval_task=RingMatch(scramble=True, batch_size=1024))

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
            Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=w), match_experiment, experiment_args=common_args),
        ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases, key_name='scramble_eval', eval_task=RingMatch(scramble=True, batch_size=1024))

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
### KNN Experimentation
n_out = 6

def compute_dists(point, data):
    diff = data - point
    dists = np.sqrt(np.diag(diff @ diff.T))
    return dists

task = RingMatch(data_size=32, n_points=n_out)
model = MlpConfig(n_layers=3, n_hidden=32, n_out=n_out)

state, hist = train(model, task, loss='ce', train_iters=5_000, test_every=1_000)
# <codecell>
full_task = RingMatch(n_points=n_out)
xs, ys = next(full_task)

logits = state.apply_fn({'params': state.params}, xs)

x = xs[0].flatten()
y = ys[0]

data = task.data[0].reshape(task.data_size, -1)

dists = compute_dists(x, data)
labs = task.data[1]

weights = np.exp(-3 * dists)
res = np.zeros(n_out)

for idx in range(n_out):
    res[idx] = np.sum(weights[labs == idx])
    
res = np.log(res + 1e-1)
plt.plot(res, 'o--')
ax = plt.twinx()
ax.plot(logits[0], 'o--', color='red')

print(labs[np.argmin(dists)])



# <codecell>
### LABEL RING MATCH TASK

# TODO: test new seed paradigm
n_iters = 3
n_out = 6

def make_common():
    return {'train_iters': 50_000, 'n_points': n_out, 'seed': new_seed()}

label_match_experiment = functools.partial(experiment, task_class=LabelRingMatch)

all_cases = []
for _ in range(n_iters):
    all_cases.extend([
        Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=512), label_match_experiment, experiment_args=make_common()),
        Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=512, use_mlp_layers=True, pos_emb=True), label_match_experiment, experiment_args=make_common()),
        Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=512), label_match_experiment, experiment_args=make_common()),
    ])

for case in tqdm(all_cases):
    print('CASE', case.name)
    print('seed', case.experiment_args['seed'])
    case.run()

## TODO: incorporate utility to plot training curves

# <codecell>
### RADIUS GENERALIZATION
radii = [0.1, 0.5, 1, 2, 4, 8]
for r in radii:
    eval_cases(all_cases, 
               [LabelRingMatch(n_points=n_out, radius=r, batch_size=1024, seed=c.experiment_args['seed']) for c in all_cases], 
               key_name=f'rad_{r}')


# <codecell>
df = pd.DataFrame(all_cases)
df_rad = pd.DataFrame(df['info'].tolist())


# <codecell>
df.to_pickle('tmp.pkl') # TODO: finish plotting <-- STOPPED HERE

# <codecell>
df = pd.read_pickle('tmp.pkl')

# <codecell>
df_rad = pd.DataFrame(df['info'].tolist())
df_rad = df_rad.rename(lambda x: x.split('_')[1], axis='columns')

plot_df = df[['name']].join(df_rad) \
                      .melt(id_vars='name', var_name='radius', value_name='acc')
plot_df['acc'] = plot_df['acc'].astype('float32')

# <codecell>
plt.gcf().set_size_inches(6, 3)
sns.barplot(plot_df, x='radius', y='acc', hue='name')

plt.tight_layout()
plt.savefig('fig/match_label_radius.png')

# <codecell>

'''
Tests to try:
- for large n_out, can a model handle small n_out? (with same number of classes)
- vary depth and compare to MNN

- eventually: probe MLP for evidence of "multiplicative" interactions
'''


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
