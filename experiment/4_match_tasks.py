"""
Experimenting with the match tasks (TODO: longer description)
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from train import train
from model.mlp import MlpConfig
from model.poly import PolyConfig
from task.match import RingMatch


n_choices = 6
task = RingMatch(n_points=n_choices, scramble=False)

# config = MlpConfig(n_out=n_choices, n_layers=3, n_hidden=128)
config = PolyConfig(n_out=n_choices, n_layers=1, n_hidden=32, start_with_dense=False, use_mult_signage=True, learnable_signage_temp=False)

state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=100_000, lr=1e-4, l1_weight=1e-4)

# %%
task_scramble = RingMatch(n_points=n_choices, scramble=True, batch_size=1024)
task_large = RingMatch(n_points=n_choices, radius=1, batch_size=1024)

xs, labs = next(task_scramble)

logits = state.apply_fn({'params': state.params}, xs)
preds = logits.argmax(axis=1)
eval_acc = np.mean(labs == preds)
eval_acc

# <codecell>

'''
Observations: (need to solidify)

- MLP learns this in-context
- Increasing radius is no problem --> performs some sort of dot product operation
- All models struggle to generalize from fixed --> scramble task
    - MNN seems to perform marginally better
    --> uses ring structure, not purely dot-product-driven

Tentative conclusion: with perfect data, in-context learning is well
within capacity of MLP (among other models)
'''
