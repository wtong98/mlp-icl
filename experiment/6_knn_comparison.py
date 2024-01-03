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


'''
Things to solidify:
- Accuracy matches (for shallow models)
    - General, scramble generalization
    - Hypothesis: KNN matched at point of perfect training accuracy <-- STOPPED HERE
- Logits match (for shallow models)
- Measure divergence for deeper models
- Compare to other tasks (e.g. Oddball)
'''

# <codecell>
# TODO: make plots v
# train_acc = [m.accuracy for m in all_cases[2].hist['train']]
# test_acc = [m.accuracy for m in all_cases[2].hist['test']]

# plt.plot(train_acc)
# plt.plot(test_acc)
# plt.axhline(y=all_cases[3].info['eval_acc'])

# <codecell>
df = pd.read_pickle('remote/6_knn_comparison/res.pkl')

def extract_plot_vals(row):
    data_size = row['data_size']
    if np.isnan(data_size):
        data_size = row['train_task'].data_size

    return pd.Series([
        row['name'],
        data_size,
        row['info']['eval_acc'].item(),
    ], index=['name', 'data_size', 'acc'])

plot_df = df.apply(extract_plot_vals, axis=1)

sns.barplot(plot_df, x='data_size', y='acc', hue='name')

# %%
import pickle

for case in all_cases:
    case.experiment = None

with open('tmp.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)