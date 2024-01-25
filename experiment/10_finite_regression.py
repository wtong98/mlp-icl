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


df = pd.read_pickle('remote/10_finite_regression/res.pkl')

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_dims,
        row['info']['eval_acc'].item(),
    ], index=['name', 'data dim', 'mse'])

plot_df = df.apply(extract_plot_vals, axis=1)

# <codecell>
task = FiniteLinearRegression(n_points=16, n_ws=128, batch_size=256, n_dims=2)
dummy_xs, _ = next(task)
dummy_xs = dummy_xs.reshape(dummy_xs.shape[0], -1)

# config = MlpConfig(n_out=1, n_layers=2, n_hidden=512)
# config = PolyConfig(n_out=1, n_layers=1, n_hidden=512, start_with_dense=True)
config = TransformerConfig(pos_emb=True, n_out=1, n_layers=4, n_hidden=512, n_mlp_layers=3, layer_norm=True, use_single_head_module=True, softmax_att=False)
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
