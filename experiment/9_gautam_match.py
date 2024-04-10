"""
Probing the abilities of each model on an exact implementation of Gautam's
in-context classification task (Reddy 2023)

Initial observations
- Transformer learns with the best sample efficiency, followed by MLP then MNN.
It seems that all models can interpolate perfectly with sufficient time
- MLPs will generalize with sufficiently many fixed clusters (just like transformers).
For small number of labels and/or points, MLPs seem to match transformers in
terms of generalizing with clusters.

To test:
- Will increasing labels or points slow the MLP's generalization as compared to the transformers?


author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>

import jax
from flax.serialization import from_state_dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from common import *

import sys
sys.path.append('../')
from train import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.match import GautamMatch

# <codecell>
df = pd.read_pickle('remote/9_gautam_match.py/res_large.pkl')
df_res = pd.DataFrame(df['info'].tolist())
df_args = pd.DataFrame(df_res['task_args'].tolist())
df_res = df_res.drop('task_args', axis='columns').join(df_args).astype('float')
df = df.drop('info', axis='columns').join(df_res)
plot_df = df[['name', 'acc', 'iwl_acc', 'icl_resamp_acc', 'icl_swap_acc', 'n_classes']] \
            .melt(id_vars=['name', 'n_classes'], var_name='acc_type', value_name='accuracy') \
            .fillna(float("inf"))

plot_df.head()

# <codecell>
sns.catplot(plot_df, x='n_classes', y='accuracy', hue='name', row='acc_type', kind='bar', height=1.5, aspect=3.5)
plt.savefig('fig/match_gautam_n_class_labs_8_pts_8_burst_4.png')

# <codecell>
### OLD res.pkl plotting  v
df = pd.read_pickle('remote/9_gautam_match.py/res.pkl')

df_res = pd.DataFrame(df['info'].tolist()).astype('float')
plot_df = df[['name']].join(df_res)

plot_df = plot_df.melt(id_vars='name', var_name='type', value_name='accuracy')
plot_df.head()

# <codecell>
g = sns.barplot(plot_df, x='type', y='accuracy', hue='name')
g.legend_.set_title(None)

plt.tight_layout()
plt.savefig('fig/match_gautam_gen.png')

# <codecell>
for i in range(6):
    row = df.iloc[i]
    accs = [m['accuracy'] for m in row['hist']['train']]
    plt.plot(accs, '--', label=row['name'], alpha=0.5)

plt.legend()
plt.xlabel('Batch (x1000)')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('fig/match_gautam_acc_curve.png')

# <codecell>

n_labels = 32

task = GautamMatch(width=1, batch_size=128, n_labels=n_labels, n_points=8, bursty=4, n_classes=1024, n_dims=8, seed=23, reset_rng_for_data=True, eps=0.1)

# config = MlpConfig(n_out=n_labels, n_layers=3, n_hidden=512)
# config = PolyConfig(n_out=n_labels, n_layers=1, n_hidden=512, start_with_dense=True)
config = TransformerConfig(pos_emb=True, n_out=n_labels, n_heads=4, n_layers=3, n_hidden=256, n_mlp_layers=3)

state, hist = train(config, data_iter=iter(task), loss='ce', test_every=500, train_iters=2000, lr=1e-4, l1_weight=1e-4)

'''

Notes:
- IWL params: bursty=1, n_classes=128
- ICL params: bursty=4, n_classes=2048
'''
# %%
task = GautamMatch(width=1, batch_size=128, n_labels=n_labels, n_points=8, bursty=4, n_classes=1024, n_dims=8, seed=23, reset_rng_for_data=True, eps=0.1)
task.batch_size = 1024

# task.matched_target = False
# task.swap_labels()
# task.resample_clusters()

xs, ys = next(task)
logits = state.apply_fn({'params': state.params}, xs)
preds = logits.argmax(-1)

np.mean(preds == ys)

# <codecell>
task = GautamMatch(width=64, batch_size=128, n_labels=n_labels, n_points=4, bursty=1, n_classes=None, n_dims=2, seed=12, reset_rng_for_data=True, eps=0)
task.batch_size = 3


xs, ys = next(task)
print('XS', xs[:,:,:2])
print('YS', ys)

logits = state.apply_fn({'params': state.params}, xs)
preds = logits.argmax(-1)
print('PREDS', preds)
np.mean(preds == ys)
# %%
for l in logits:
    plt.plot(l)
# %%
xs[1] @ xs[1].T
# %%
