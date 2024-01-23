"""
Testing transition to ICL in MLPs and Transformers (and MNNs, perhaps), using
the construction given in Raventos et al. 2023

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
from task.regression import FiniteLinearRegression

def t(xs):
      return np.swapaxes(xs, -2, -1)


# TODO: test
def uninterleave(interl_xs):
      xs = interl_xs[:,0::2]
      ys = interl_xs[:,1::2,[0]]
      xs, x_q = xs[:,:-1], xs[:,[-1]]
      return xs, ys, x_q


def estimate_dmmse(xs, ys, x_q, ws, sig=0.5):
      '''
      xs: N x P x D
      ys: N x P x 1
      x_q: N x 1 x D
      ws: F x D
      '''
      
      weights = np.exp(-(1 / (2 * sig**2)) * np.sum((ys - xs @ ws.T)**2, axis=1))  # N x F
      probs = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-32)
      w_dmmse = np.expand_dims(probs, axis=-1) * ws  # N x F x D
      w_dmmse = np.sum(w_dmmse, axis=1, keepdims=True)  # N x 1 x D
      return (x_q @ t(w_dmmse)).squeeze()


def estimate_ridge(xs, ys, x_q, sig=0.01):
      n_dims = xs.shape[-1]
      w_ridge = np.linalg.pinv(t(xs) @ xs + sig**2 * np.identity(n_dims)) @ t(xs) @ ys
      return (x_q @ w_ridge).squeeze()



n_dims = 2

task = FiniteLinearRegression(n_ws=8, n_points=16, batch_size=256, n_dims=n_dims, seed=None, reset_rng_for_data=False)
xs, y_q = next(task)
xs, ys, x_q = uninterleave(xs)
preds = estimate_dmmse(xs, ys, x_q, task.ws)
# preds = estimate_ridge(xs, ys, x_q)

print(np.mean((preds - y_q)**2))

new_task = FiniteLinearRegression(n_points=16, n_dims=n_dims, n_ws=None)
xs_test, y_q_test = next(new_task)
xs_test, ys_test, x_q_test = uninterleave(xs_test)
preds_test = estimate_dmmse(xs_test, ys_test, x_q_test, task.ws)
# preds_test = estimate_ridge(xs_test, ys_test, x_q_test)

np.mean((preds_test - y_q_test)**2)

# <codecell>
task = FiniteLinearRegression(n_ws=8, batch_size=256, n_dims=2)
config = MlpConfig(n_out=1, n_layers=3, n_hidden=512)
# config = PolyConfig(n_out=1, n_layers=1, n_hidden=512, start_with_dense=True)
# config = TransformerConfig(pos_emb=True, n_out=1, n_heads=4, n_layers=4, n_hidden=128, n_mlp_layers=3)

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
