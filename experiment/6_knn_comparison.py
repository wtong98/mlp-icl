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
@ dataclass
class KnnCase:
    name: str
    config: KnnConfig
    task_class: Callable
    data_size: int
    seed: int
    task_args: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def run(self):
        self.task = self.task_class(data_size=self.data_size, seed=self.seed, **self.task_args)
        self.config.xs = self.task.data[0].reshape(self.data_size, -1)
        self.config.ys = self.task.data[1]
        self.model = self.config.to_model()

    def eval(self, task, key_name='eval_acc'):
        xs, ys = next(task)
        probs = self.model(xs)
        eval_acc = np.mean(ys == probs.argmax(-1))
        self.info[key_name] = eval_acc
    

n_iters = 1
n_out = 6
data_sizes = [8, 16, 32, 64, 128, 256, 512]
# data_sizes = [128, 256, 512]
# train_iters=20_000
train_iters=8_000

all_cases = []
for _ in range(n_iters):
    for size in data_sizes:

        common_seed = new_seed()

        def make_common():
            return {'train_iters': train_iters, 'data_size': size, 'seed': common_seed, 'reset_rng_for_data': True}

        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_out, n_layers=1, n_hidden=256), match_experiment, experiment_args=make_common()),
            # Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=256, use_mlp_layers=True, pos_emb=True), match_experiment, experiment_args=make_common()),
            # Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=256), match_experiment, experiment_args=make_common()),

            KnnCase('KNN', KnnConfig(beta=99, n_classes=n_out), task_class=RingMatch, data_size=size, seed=common_seed)
        ])

for case in tqdm(all_cases):
    case.run()


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
task = RingMatch(n_points=n_out, batch_size=1024)
eval_cases(all_cases, task)

# <codecell>
df = pd.DataFrame(all_cases)

def extract_plot_vals(row):
    data_size = row['data_size']
    if np.isnan(data_size):
        data_size = row['experiment_args']['data_size']

    return pd.Series([
        row['name'],
        data_size,
        row['info']['eval_acc'].item(),
    ], index=['name', 'data_size', 'acc'])

plot_df = df.apply(extract_plot_vals, axis=1)
# plot_df

sns.barplot(plot_df, x='data_size', y='acc', hue='name')

# %%
import pickle

for case in all_cases:
    case.experiment = None

with open('tmp.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)