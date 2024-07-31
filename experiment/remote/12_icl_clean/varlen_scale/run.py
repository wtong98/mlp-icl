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
from model.mlp import MlpConfig , SpatialMlpConfig
from model.transformer import TransformerConfig
from task.regression import FiniteLinearRegression 

run_id = new_seed()
print('RUN ID', run_id)


batch_size = 64

run_split = 9

train_iters_mlp = 2_048_000
depths_mlp = [2, 4, 8]
widths_mlp = [128, 512, 2048]

train_iters_mix = 500_000
depths_mix = [2, 4, 8]
widths_mix = [32, 128, 512]
channels_mix = 64

train_iters_trans = 600_000
depths_trans = [2, 4, 8]
widths_trans = [32, 128, 512]

n_dims = 8
n_points = 8
n_ws_set = [None]

### START TEST CONFIGS
# run_split = 1

# train_iters_mlp = 2_000
# depths_mlp = [2]
# widths_mlp = [128]

# train_iters_mix = 2_000
# depths_mix = [2]
# widths_mix = [64]
# channels_mix = 4

# train_iters_trans = 2_000
# depths_trans = [2]
# widths_trans = [32]

# n_dims = 8
# n_points = 8
# n_ws_set = [None]
### END TEST CONFIGS


all_cases = []

for n_ws in n_ws_set:
    for depth in depths_mlp:
        for width in widths_mlp:
            common_task_args = {'var_length': True, 'n_ws': n_ws, 'n_dims': n_dims, 'n_points': n_points, 'seed': new_seed()}

            all_cases.append(
                Case('MLP', MlpConfig(n_out=1, n_layers=depth, n_hidden=width),
                    train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                    train_task = FiniteLinearRegression(batch_size=batch_size, **common_task_args),
                    test_task=FiniteLinearRegression(batch_size=1024, **common_task_args),
                    info={'common_task_args': common_task_args})
            )

    for depth in depths_mix:
        for width in widths_mix:
            common_task_args = {'var_length': True, 'n_ws': n_ws, 'n_dims': n_dims, 'n_points': n_points, 'seed': new_seed()}

            all_cases.append(
                Case('Mixer', SpatialMlpConfig(n_out=1, n_layers=depth, n_hidden=width, n_channels=channels_mix),
                    train_args={'train_iters': train_iters_mix, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                    train_task = FiniteLinearRegression(batch_size=batch_size, **common_task_args),
                    test_task=FiniteLinearRegression(batch_size=1024, **common_task_args),
                    info={'common_task_args': common_task_args})
            )


    for depth in depths_trans:
        for width in widths_trans:
            common_task_args = {'var_length': True, 'n_ws': n_ws, 'n_dims': n_dims, 'n_points': n_points, 'seed': new_seed()}

            all_cases.append(
                Case('Transformer', TransformerConfig(n_out=1, n_layers=depth, n_hidden=width, pos_emb=False, n_heads=1, n_mlp_layers=2),
                    train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                    train_task = FiniteLinearRegression(batch_size=batch_size, **common_task_args),
                    test_task=FiniteLinearRegression(batch_size=1024, **common_task_args),
                    info={'common_task_args': common_task_args})
            )

all_cases = split_cases(all_cases, run_split)
print('CURRENT CASES:', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, tasks, key_name='mse_pretrain', use_mse=True)

true_tasks = []
for c in all_cases:
    task_args = c.info['common_task_args'].copy()
    task_args['n_ws'] = None
    true_tasks.append(FiniteLinearRegression(batch_size=1024, **task_args))
eval_cases(all_cases, true_tasks, key_name='mse_true', use_mse=True)

for case in all_cases:
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.info['flops'] = case.get_flops()
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')
