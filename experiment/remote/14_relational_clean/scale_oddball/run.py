"""
Observing scaling laws associated with model size and training iterations
"""

# <codecell>
import jax
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, DotMlpConfig
from model.transformer import TransformerConfig
from task.oddball import FreeOddballTask, LineOddballTask

run_id = new_seed()
print('RUN ID', run_id)

batch_size = 128

train_iters_dot = 64_000

train_iters_mlp = 8_000
depths_mlp = [1, 2, 4]
widths_mlp = [4, 16, 64, 256]

train_iters_trans = 8_000
depths_trans = [1, 2, 4]
widths_trans = [8, 32]

n_points = 6

### START TEST CONFIGS
# train_iters_dot = 64_0

# train_iters_mlp = 8_0
# depths_mlp = [1]
# widths_mlp = [4]

# train_iters_trans = 8
# depths_trans = [1]
# widths_trans = [8]
### END TEST CONFIGS

all_cases = []
common_args = {'n_choices': n_points}

for depth in depths_mlp:
    for width in widths_mlp:
        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_points, n_layers=depth, n_hidden=width),
                train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = FreeOddballTask(batch_size=batch_size, **common_args),
                test_task=FreeOddballTask(batch_size=1024, **common_args)),

            Case('MLP', MlpConfig(n_out=n_points, n_layers=depth, n_hidden=width),
                train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = LineOddballTask(batch_size=batch_size, **common_args),
                test_task=LineOddballTask(batch_size=1024, **common_args)),
        ])


for depth in depths_trans:
    for width in widths_trans:
        all_cases.extend([
            Case('Transformer', TransformerConfig(n_out=n_points, n_layers=depth, n_hidden=width, pos_emb=True, n_mlp_layers=2),
                train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = FreeOddballTask(batch_size=batch_size, **common_args),
                test_task=FreeOddballTask(batch_size=1024, **common_args)),

            Case('Transformer', TransformerConfig(n_out=n_points, n_layers=depth, n_hidden=width, pos_emb=True, n_mlp_layers=2),
                train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = LineOddballTask(batch_size=batch_size, **common_args),
                test_task=LineOddballTask(batch_size=1024, **common_args))
        ])

all_cases.extend([
    Case('RB MLP', 
        DotMlpConfig(n_out=n_points, use_initial_proj=False, last_token_only=False),
        train_args={'train_iters': train_iters_dot, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
        train_task = FreeOddballTask(batch_size=batch_size, **common_args),
        test_task=FreeOddballTask(batch_size=1024, **common_args)),

    Case('RB MLP', 
        DotMlpConfig(n_out=n_points, use_initial_proj=False, last_token_only=False),
        train_args={'train_iters': train_iters_dot, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
        train_task = LineOddballTask(batch_size=batch_size, **common_args),
        test_task=LineOddballTask(batch_size=1024, **common_args)),
])


for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=test_tasks, key_name='acc')

for case in all_cases:
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.info['flops'] = case.get_flops()
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
