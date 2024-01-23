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


# <codecell>
task = FiniteLinearRegression(n_ws=128, batch_size=256, n_dims=2)
# config = MlpConfig(n_out=1, n_layers=3, n_hidden=512)
# config = PolyConfig(n_out=1, n_layers=1, n_hidden=512, start_with_dense=True)
config = TransformerConfig(pos_emb=True, n_out=1, n_heads=4, n_layers=3, n_hidden=128, n_mlp_layers=3)

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
