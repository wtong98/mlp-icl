"""
Comparing behavior of models versus KNN (on finite data sizes), at the
accuracy and logit level.
"""

# <codecell>
import functools

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
from task.match import RingMatch, LabelRingMatch

match_experiment = functools.partial(experiment, task_class=RingMatch)

# <codecell>
n_iters = 3
n_out = 6
data_sizes = [16, 32, 64, 128, 256, 512, 1024]

all_cases = []
for _ in range(n_iters):
    for size in data_sizes:

        common_args = {'train_iters': 20_000, 'data_size': size}

        # TODO: save task <-- STOPPED HERE
        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=256), match_experiment, experiment_args=common_args),
            Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=256, use_mlp_layers=True, pos_emb=True), match_experiment, experiment_args=common_args),
            Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=256), match_experiment, experiment_args=common_args),
        ])

for case in tqdm(all_cases):
    case.run()


'''
Things to solidify:
- Accuracy matches (for shallow models)
    - General, scramble generalization
- Logits match (for shallow models)
- Measure divergence for deeper models
- Compare to other tasks (e.g. Oddball)
'''
# %%
import pickle

for case in all_cases:
    case.experiment = None

with open('tmp.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)