import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../')
sys.path.append('../../../')

from common import *
from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from task.match import LabelRingMatch

# TESTING VARIOUS DEPTH
n_iters = 3
n_out = 6
depths = [1, 2, 3, 4, 6, 8, 12]

train_args = {'train_iters': 50_000}

all_cases = []
for _ in range(n_iters):
    for d in depths:
        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_out, n_layers=d, n_hidden=512), train_args=train_args),
            Case('Transformer', TransformerConfig(n_out=n_out, n_layers=d, n_hidden=512, use_mlp_layers=True, pos_emb=True), train_args=train_args),
        ])

for case in tqdm(all_cases):
    case.train_task = LabelRingMatch(n_points=n_out, seed=new_seed(), reset_rng_for_data=True)
    case.run()

# TODO: radius and partial tests
eval_cases(all_cases, eval_task=case.train_task, batch_size=1024)

for case in all_cases:
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle('res.pkl')
