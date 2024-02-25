"""
Searching for tasks where MLP > transformer

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>

from common import *

import sys
sys.path.append('../')
from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from task.function import LinearTask


task = LinearTask(n_dims=64, seed=5, tokenize=True)
# config = MlpConfig(n_out=1, n_layers=3, n_hidden=128, act_fn='relu')
config = TransformerConfig(pos_emb=True, n_out=1, n_layers=3, n_heads=2, n_hidden=256, n_mlp_layers=3, layer_norm=True, max_len=128)

state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=10_000, lr=1e-4)
# %%
