"""
Observing scaling laws associated with model size and training iterations
"""

# <codecell>
import jax
from flax.serialization import to_state_dict
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig 
from model.transformer import TransformerConfig
from task.regression import FiniteLinearRegression 


run_id = new_seed()
print('RUN ID', run_id)


batch_size = 128

train_iters_mlp = [1000, 4000, 16_000, 64_000, 256_000]
depths_mlp = [1, 2, 4]
widths_mlp = [32, 128, 512]

train_iters_trans = [250, 1000, 4000, 16_000, 64_000]
depths_trans = [1, 2, 4]
widths_trans = [32, 128, 512]

n_dims = 8
n_points = 16
n_ws = None

all_cases = []

for train_iters in train_iters_mlp:
    for depth in depths_mlp:
        for width in widths_mlp:
            common_task_args = {'n_ws': n_ws, 'n_dims': n_dims, 'n_points': n_points, 'seed': new_seed()}
            train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'}

            all_cases.append(
                Case('MLP', MlpConfig(n_out=1, n_layers=depth, n_hidden=width),
                     train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                     train_task = FiniteLinearRegression(batch_size=batch_size, **common_task_args),
                     test_task=FiniteLinearRegression(batch_size=1024, **common_task_args))
            )


for train_iters in train_iters_trans:
    for depth in depths_trans:
        for width in widths_trans:
            common_task_args = {'n_ws': n_ws, 'n_dims': n_dims, 'n_points': n_points, 'seed': new_seed()}
            train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'}

            all_cases.append(
                Case('Transformer', TransformerConfig(n_out=1, n_layers=depth, n_hidden=width, pos_emb=False, n_heads=2, n_mlp_layers=3),
                     train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                     train_task = FiniteLinearRegression(batch_size=batch_size, **common_task_args),
                     test_task=FiniteLinearRegression(batch_size=1024, **common_task_args))
            )

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, tasks, key_name='eval_mse', use_mse=True)

for case in all_cases:
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

