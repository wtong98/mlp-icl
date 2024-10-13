"""
Observing scaling laws associated with model size and training iterations
"""

# <codecell>
import jax
import pandas as pd
from tqdm import tqdm

import sys


from common import *
from mlp_icl.model.mlp import MlpConfig
from mlp_icl.model.transformer import TransformerConfig
from mlp_icl.task.function import PowerTask 

run_id = new_seed()
print('RUN ID', run_id)

batch_size = 128

train_iters_mlp = 64_000
depths_mlp = [1, 2, 4]
widths_mlp = [4, 16, 64, 256]

train_iters_trans = 256_000
depths_trans = [1, 2, 4]
widths_trans = [8, 32]

# n_dims = [2, 4, 8, 16, 32, 64]
n_dims = [64]
powers = [1, 2, 3]
powers = [1]

### START TEST CONFIGS
# train_iters_mlp = 64
# depths_mlp = [1]
# widths_mlp = [4]

# train_iters_trans = 256
# depths_trans = [1]
# widths_trans = [8]

# n_dims = [2]
# powers = [1]
### END TEST CONFIGS

all_cases = []

for n_d in n_dims:
    for depth in depths_mlp:
        for width in widths_mlp:
            for p in powers:
                common_args = {'n_dims': n_d, 'seed': new_seed(), 'power': p}

                all_cases.append(
                    Case('MLP', MlpConfig(n_out=1, n_layers=depth, n_hidden=width),
                        train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                        train_task = PowerTask(batch_size=batch_size, **common_args),
                        test_task=PowerTask(batch_size=1024, **common_args))
                )


    for depth in depths_trans:
        for width in widths_trans:
            for p in powers:
                common_args = {'n_dims': n_d, 'tokenize': 1, 'seed': new_seed(), 'power': p}

                all_cases.append(
                    Case('Transformer', TransformerConfig(n_out=1, n_layers=depth, n_hidden=width, pos_emb=True, n_mlp_layers=2),
                        train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'},
                        train_task = PowerTask(batch_size=batch_size, **common_args),
                        test_task=PowerTask(batch_size=1024, **common_args))
                )


for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=test_tasks, key_name='mse', use_mse=True)

for case in all_cases:
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.info['flops'] = case.get_flops()
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
