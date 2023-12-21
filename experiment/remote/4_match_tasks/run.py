import functools
import pickle

from tqdm import tqdm

from common import *

import sys
sys.path.append('../../../')
from train import train
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.match import RingMatch, LabelRingMatch

# TESTING VARIOUS DEPTH
n_iters = 3
n_out = 6
depths = [1, 2, 3, 4, 6, 8, 12]

match_experiment = functools.partial(experiment, task_class=RingMatch)

common_args = {'train_iters': 50_000}

all_cases = []
for _ in range(n_iters):
    for d in depths:
        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_out, n_layers=d, n_hidden=512), match_experiment, experiment_args=common_args),
            Case('Transformer', TransformerConfig(n_out=n_out, n_layers=d, n_hidden=512, use_mlp_layers=True, pos_emb=True), match_experiment, experiment_args=common_args),
        ])

for case in tqdm(all_cases):
    case.run()

with open('res.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

