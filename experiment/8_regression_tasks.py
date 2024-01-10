"""
Can these models learn regression in-context? We're about to find out!

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from common import *

import sys
sys.path.append('../')
from train import train
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegression

'''
Things to try:
- Far out generalization to OOD x, w
- Single probes on cross-sections of x
- comparison to KNN
'''

task = LinearRegression(batch_size=128)
config = MlpConfig(n_out=1, n_layers=3, n_hidden=256)
# config = PolyConfig(n_out=1, n_layers=1, n_hidden=512, start_with_dense=True)
# config = TransformerConfig(pos_emb=True, n_out=1, n_heads=4, n_layers=6, n_hidden=512, use_mlp_layers=False)

state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=2_000, lr=1e-4, l1_weight=1e-4)


# <codecell>
xs = np.array([[1, 0], [1, 0], 
               [0, 1], [1, 0],
               [1, 1], [1, 0],
               [-1, -1], [-2, 0],
               [-1, 0], [-1, 0],
               [10, 0]])

xs = np.expand_dims(xs, axis=0)
state.apply_fn({'params': state.params}, xs)

# <codecell>

from flax.serialization import from_state_dict, to_state_dict

from_state_dict(state, to_state_dict(state)).apply_fn
