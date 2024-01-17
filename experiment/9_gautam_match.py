"""
Probing the abilities of each model on an exact implementation of Gautam's
in-context classification task (Reddy 2023)

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
df = pd.read_pickle('remote/9_gautam_match.py/res.pkl')

df_res = pd.DataFrame(df['info'].tolist()).astype('float')
plot_df = df[['name']].join(df_res)

plot_df = plot_df.melt(id_vars='name', var_name='type', value_name='acc')
plot_df.head()

# <codecell>
g = sns.barplot(plot_df, x='type', y='acc', hue='name')
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

task = GautamMatch(batch_size=128, n_labels=n_labels, bursty=1, n_classes=128, n_dims=64, seed=5)

config = MlpConfig(n_out=n_labels, n_layers=3, n_hidden=512)
# config = PolyConfig(n_out=n_labels, n_layers=1, n_hidden=512, start_with_dense=True)
# config = TransformerConfig(pos_emb=True, n_out=n_labels, n_heads=4, n_layers=3, n_hidden=256, n_mlp_layers=3)

state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=5_000, lr=1e-4, l1_weight=1e-4)

'''
Initial observations
- Transformer learns with the best sample efficiency, followed by MLP then MNN.
It seems that all models can interpolate perfectly with sufficient time

Notes:
- IWL params: bursty=1, n_classes=128
- ICL params: bursty=4, n_classes=2048
'''
# %%
task = GautamMatch(batch_size=128, n_labels=n_labels, bursty=1, n_classes=128, n_dims=64, seed=5)
task.batch_size = 1024

# task.matched_target = False
task.swap_labels()

xs, ys = next(task)
logits = state.apply_fn({'params': state.params}, xs)
preds = logits.argmax(-1)

np.mean(preds == ys)