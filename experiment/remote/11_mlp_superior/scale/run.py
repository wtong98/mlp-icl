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
from task.function import LinearTask 


run_id = new_seed()
print('RUN ID', run_id)

batch_size = 128

train_iters_mlp = [125, 256, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
depths_mlp = [1, 2, 4]
widths_mlp = [16, 32, 64, 128, 256]

train_iters_trans = [500, 2_000, 8_000, 32_000]
depths_trans = [1, 2, 3, 4, 6, 8]
widths_trans = [64, 128]

n_dims = 64
all_cases = []

for train_iters in train_iters_mlp:
    for depth in depths_mlp:
        for width in widths_mlp:
            common_args = {'n_dims': n_dims, 'seed': new_seed()}

            all_cases.append(
                Case('MLP', MlpConfig(n_out=1, n_layers=depth, n_hidden=width),
                    train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                    train_task = LinearTask(batch_size=batch_size, **common_args),
                    test_task=LinearTask(batch_size=1024, **common_args))
            )


for train_iters in train_iters_trans:
    for depth in depths_trans:
        for width in widths_trans:
            common_args = {'n_dims': n_dims, 'tokenize': True, 'seed': new_seed()}

            all_cases.append(
                Case('Transformer', TransformerConfig(n_out=1, n_layers=depth, n_hidden=width, pos_emb=True, n_mlp_layers=2),
                    train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                    train_task = LinearTask(batch_size=batch_size, **common_args),
                    test_task=LinearTask(batch_size=1024, **common_args))
            )

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()


test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=test_tasks, key_name='mse', use_mse=True)


for case in all_cases:
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

