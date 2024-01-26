"""
Testing transition to ICL in MLPs and Transformers (and MNNs, perhaps), using
the construction given in Raventos et al. 2023

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import jax
import jax.numpy as jnp
import flax.linen as nn
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
from model.mlp import MlpConfig, RfConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import FiniteLinearRegression, LinearRegression

# dummies
def estimate_dmmse():
    pass

def estimate_ridge():
    pass

# <codecell>
### PLOTTING MLP CURVES
df = pd.read_pickle('remote/10_finite_regression/res_mlp.pkl')
len(df.iloc[-1].train_task.ws) or None

#<codecell>

def extract_plot_vals(row):
    n_ws = row.train_task.ws
    if n_ws is not None:
        n_ws = len(n_ws)
    else:
        n_ws = float('inf')
    
    return pd.Series([
        row['name'],
        n_ws,
        row['info']['mse_pretrain'].item(),
        row['info']['mse_true'].item(),
    ], index=['name', 'n_betas', 'mse_pretrain', 'mse_true'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .melt(id_vars=['name', 'n_betas'], var_name='mse_type', value_name='mse')

plot_df

# <codecell>
g = sns.catplot(plot_df, x='n_betas', y='mse', hue='name', row='mse_type', kind='point')
[ax.set_yscale('log') for ax in g.axes.ravel()]
g.figure.set_size_inches(8, 6)
plt.savefig('fig/reg_finite_mlp_dim2.png')


# <codecell>
### PLOTTING REP'd CURVES
pkl_path = Path('remote/10_finite_regression/rep')
dfs = [pd.read_pickle(f) for f in pkl_path.iterdir() if f.suffix == '.pkl']
df = pd.concat(dfs)

def extract_plot_vals(row):
    n_ws = row.train_task.ws
    if n_ws is not None:
        n_ws = len(n_ws)
    else:
        n_ws = float('inf')
    
    return pd.Series([
        row['name'],
        n_ws,
        row['info']['mse_pretrain'].item(),
        row['info']['mse_true'].item(),
    ], index=['name', 'n_betas', 'mse_pretrain', 'mse_true'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .melt(id_vars=['name', 'n_betas'], var_name='mse_type', value_name='mse')
plot_df

# <codecell>
g = sns.catplot(plot_df, x='n_betas', y='mse', hue='name', row='mse_type', kind='point')
[ax.set_yscale('log') for ax in g.axes.ravel()]
g.figure.set_size_inches(8, 6)
plt.savefig('fig/reg_finite_dim2_finer.png')



# <codecell>
'''
TODO: rather than larger dimension spaces, may want to consider finer resolution with n_dims=2
TODO: plot loss curves
Hypothesis: the sample complexity of an MLP scales extremely poorly with dimensionality
'''


task = FiniteLinearRegression(n_points=16, n_ws=128, batch_size=256, n_dims=4)
dummy_xs, _ = next(task)
dummy_xs = dummy_xs.reshape(dummy_xs.shape[0], -1)

# config = MlpConfig(n_out=1, n_layers=3, n_hidden=512)
# config = PolyConfig(n_out=1, n_layers=1, n_hidden=512, start_with_dense=True)
# config = TransformerConfig(pos_emb=True, n_out=1, n_layers=4, n_hidden=512, n_mlp_layers=3, layer_norm=True, use_single_head_module=True, softmax_att=False)
config = TransformerConfig(pos_emb=True, n_out=1, n_layers=4, n_heads=2, n_hidden=512, n_mlp_layers=3, layer_norm=True)
# config = RfConfig(n_in=dummy_xs.shape[1], n_out=1, scale=1, n_hidden=512, use_quadratic_activation=True)

state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=100_000, lr=1e-4, l1_weight=1e-4)

# <codecell>
from optax import squared_error
task = FiniteLinearRegression(n_ws=None, batch_size=1024, n_dims=2)

xs, ys = next(task)
preds = state.apply_fn({'params': state.params}, xs)

print(np.mean((preds - ys)**2))
squared_error(preds, ys).shape



# <codecell>
xs = np.array([[1, 0], [1, 0], 
               [0, 1], [1, 0],
               [1, 1], [1, 0],
               [-1, -1], [-2, 0],
               [-1, 0], [-1, 0],
               [5, -1]])

xs = np.expand_dims(xs, axis=0)
state.apply_fn({'params': state.params}, xs)

# <codecell>

from flax.serialization import from_state_dict, to_state_dict

from_state_dict(state, to_state_dict(state)).apply_fn
