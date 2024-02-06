"""
Testing transition to ICL in MLPs and Transformers (and MNNs, perhaps), using
the construction given in Raventos et al. 2023


Early observations:
- MLP transition to ICL seems to scale inversely with depth --> capacity limit pushes generalization
- MLP (1-layer and deep, also linear transformer) converge to same suboptimal MSE
    - where does this MSE come from? (roughly half of null-model MSE (mse = n_dims))
- KNN solution does not work beyond chance level

TODO: plot loss curves --> confound with sample complexity and true expressivity
TODO: examine pattern of errors --> guesses at least the right sign?
Hypothesis: the sample complexity of an MLP scales extremely poorly with dimensionality

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import flax.linen as nn
from flax.serialization import from_state_dict
import optax
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
plt.savefig('fig/reg_finite_mlp_dim8.png')


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
        row.train_task.n_dims,
        row['info']['mse_pretrain'].item(),
        row['info']['mse_true'].item(),
    ], index=['name', 'n_betas', 'n_dims', 'mse_pretrain', 'mse_true'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .melt(id_vars=['name', 'n_betas', 'n_dims'], var_name='mse_type', value_name='mse')
plot_df

# <codecell>
g = sns.catplot(plot_df[plot_df.n_dims == 8], x='n_betas', y='mse', hue='name', row='mse_type', kind='point')
[ax.set_yscale('log') for ax in g.axes.ravel()]
g.figure.set_size_inches(12, 6)
plt.savefig('fig/reg_finite_dim8.png')


# <codecell>
### COMPARISON TO KNN SOLUTIONS
### result does not work beyond chance
nd = 1

task = Finite(FiniteLinearRegression(n_ws=None, n_dims=nd), data_size=2048)
c = KnnCase('test', KnnConfig(beta=7), train_task=task)

c.run()

test_task = FiniteLinearRegression(n_ws=None, batch_size=256, n_dims=nd)
c.eval_mse(test_task)
c.info['eval_mse']


# <codecell>
### COMPARISON TO IDENTITY APPROX LIN REG MODEL
### result: abject failure. Linear covariance is a very poor approximation.
### model still learns some significant covariance structure

def t(x):
    return np.swapaxes(x, -2, -1)

task = FiniteLinearRegression(n_ws=None, n_dims=64, noise_scale=0.5, enforce_orth_x=False)
xs, ys = next(task)

x = xs[:,:-1,:-1]
y = xs[:,:-1,[-1]]
x_q = xs[:,[-1],:-1]

# inv_cov = np.linalg.pinv(t(x) @ x)
inv_cov = np.identity(x.shape[-1])

pred = (x_q @ (inv_cov @ t(x) @ y)).squeeze()

np.mean((ys - pred)**2)




# <codecell>
### PLOTTING SCALE CURVES
pkl_path = Path('remote/10_finite_regression/scale')
dfs = [pd.read_pickle(f) for f in pkl_path.iterdir() if 'dimwise' in f.name and f.suffix == '.pkl']
df = pd.concat(dfs)

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row.train_task.n_dims,
        row['info']['mse_pretrain'].item(),
        row['info']['mse_true'].item(),
    ], index=['name', 'n_dims', 'mse_pretrain', 'mse_true'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .melt(id_vars=['name', 'n_dims'], var_name='mse_type', value_name='mse')
plot_df

# <codecell>
dims = 2**np.arange(10)

g = sns.catplot(plot_df, x='n_dims', y='mse', hue='name', row='mse_type', kind='point')
[ax.plot(np.arange(10), dims + 0.25, '--', color='magenta', label='Null') for ax in g.axes.ravel()]
[ax.set_yscale('log') for ax in g.axes.ravel()]
g.figure.set_size_inches(8, 6)

plt.savefig('fig/reg_finite_dim_scale_icl.png')

# <codecell>
pkl_path = Path('remote/10_finite_regression/scale')
dfs = [pd.read_pickle(f) for f in pkl_path.iterdir() if 'pointwise' in f.name and f.suffix == '.pkl']
df = pd.concat(dfs)

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row.train_task.n_points,
        row['info']['mse_pretrain'].item(),
        row['info']['mse_true'].item(),
    ], index=['name', 'n_points', 'mse_pretrain', 'mse_true'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .melt(id_vars=['name', 'n_points'], var_name='mse_type', value_name='mse')
plot_df

# <codecell>
g = sns.catplot(plot_df, x='n_points', y='mse', hue='name', row='mse_type', kind='point')
[ax.plot(np.arange(9),  np.ones(9) * 8.25, '--', color='magenta', label='Null') for ax in g.axes.ravel()]
[ax.set_yscale('log') for ax in g.axes.ravel()]
g.figure.set_size_inches(8, 6)
plt.savefig('fig/reg_finite_points_scale_icl.png')



# <codecell>
# PLOT LOSSES
for i in range(len(df)):
    row = df.iloc[-i]
    if not pd.isna(row['hist']):
        accs = [m['loss'] for m in row['hist']['train']]
        plt.plot(accs, '--', label=row['name'], alpha=0.5)

plt.legend()
plt.xlabel('Batch (x1000)')
plt.ylabel('MSE')

# plt.yscale('log')
# plt.xscale('log')

plt.tight_layout()
# plt.savefig('fig/reg_finite_dim4_curve_n_ws_2.png')


# <codecell>
### TRAINING PLAYGROUND
task = FiniteLinearRegression(n_points=16, n_ws=None, batch_size=128, n_dims=8, enforce_orth_x=True, noise_scale=0)

# TODO: check if MSE on identity-covariance model matches <-- STOPPED HERE
config = MlpConfig(n_out=1, n_layers=3, n_hidden=1024, act_fn='relu')
# config = MlpConfig(n_out=1, n_layers=1, n_hidden=4096, act_fn='gelu')
# config = PolyConfig(n_out=1, n_layers=1, n_hidden=512, start_with_dense=True)
# config = TransformerConfig(use_last_index_output=True, pos_emb=False, n_out=1, n_layers=1, n_hidden=512, n_mlp_layers=0, layer_norm=False, use_single_head_module=True, softmax_att=False)
# config = TransformerConfig(use_last_index_output=True, pos_emb=False, n_out=1, n_layers=1, pure_linear_self_att=True)
# config = TransformerConfig(pos_emb=False, n_out=1, n_layers=3, n_heads=2, n_hidden=512, n_mlp_layers=3, layer_norm=True)

state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=500_000, lr=1e-4)

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
