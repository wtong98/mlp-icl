"""
Match task accuracies
"""

# <codecell>
from flax.serialization import to_state_dict
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

train_iters_mlp = 16_000
depth_mlp = 4
width_mlp = 256

train_iters_trans = 16_000
depth_trans = 4
width_trans = 64

fo_train_dist = 5
fo_test_dists = [1, 2.5, 5, 10, 20]

lo_dists = [0.25, 0.5, 1, 2, 4, 8]

n_points = 6


### START TEST CONFIGS
# train_iters_dot = 64_0

# train_iters_mlp = 16_0
# depth_mlp = 4
# width_mlp = 256

# train_iters_trans = 16_0
# depth_trans = 4
# width_trans = 64

# fo_train_dist = 5
# fo_test_dists = [1]

# lo_dists = [0.25]
### END TEST CONFIGS

all_fo_cases = []
all_lo_cases = []

common_args = {'n_choices': n_points, 'discrim_dist': fo_train_dist}

all_fo_cases.extend([
    Case('MLP', MlpConfig(n_out=n_points, n_layers=depth_mlp, n_hidden=width_mlp),
        train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
        train_task = FreeOddballTask(batch_size=batch_size, **common_args),
        test_task=FreeOddballTask(batch_size=1024, **common_args),
        info={'common_args': common_args}),

    Case('Transformer', TransformerConfig(n_out=n_points, n_layers=depth_trans, n_hidden=width_trans, pos_emb=True, n_mlp_layers=2),
        train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
        train_task = FreeOddballTask(batch_size=batch_size, **common_args),
        test_task=FreeOddballTask(batch_size=1024, **common_args),
        info={'common_args': common_args}),

    Case('RB MLP', 
        DotMlpConfig(n_out=n_points, use_initial_proj=False, last_token_only=False),
        train_args={'train_iters': train_iters_dot, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
        train_task = FreeOddballTask(batch_size=batch_size, **common_args),
        test_task=FreeOddballTask(batch_size=1024, **common_args),
        info={'common_args': common_args})
])

for case in tqdm(all_fo_cases):
    print('RUNNING', case.name)
    case.run()

for d in fo_test_dists:
    test_task = FreeOddballTask(batch_size=1024, n_choices=n_points, discrim_dist=d)
    eval_cases(all_fo_cases, eval_task=test_task, key_name=f'acc_fo_{d}')


################
### LO CASES ###
################

for d in lo_dists:
    common_args = {'n_choices': n_points, 'perp_dist': d, 'linear_dist': 1}

    all_lo_cases.extend([
        Case('MLP', MlpConfig(n_out=n_points, n_layers=depth_mlp, n_hidden=width_mlp),
            train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
            train_task = LineOddballTask(batch_size=batch_size, **common_args),
            test_task=LineOddballTask(batch_size=1024, **common_args),
            info={'common_args': common_args}),

        Case('Transformer', TransformerConfig(n_out=n_points, n_layers=depth_trans, n_hidden=width_trans, pos_emb=True, n_mlp_layers=2),
            train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
            train_task = LineOddballTask(batch_size=batch_size, **common_args),
            test_task=LineOddballTask(batch_size=1024, **common_args),
            info={'common_args': common_args}),

        Case('RB MLP', 
            DotMlpConfig(n_out=n_points, use_initial_proj=False, last_token_only=False),
            train_args={'train_iters': train_iters_dot, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
            train_task = LineOddballTask(batch_size=batch_size, **common_args),
            test_task=LineOddballTask(batch_size=1024, **common_args),
            info={'common_args': common_args})
    ])

for case in tqdm(all_lo_cases):
    print('RUNNING', case.name)
    case.run()

for d in lo_dists:
    test_task = LineOddballTask(batch_size=1024, n_choices=n_points, linear_dist=1, perp_dist=d)
    eval_cases(all_lo_cases, eval_task=test_task, key_name=f'acc_lo_{d}')


all_cases = all_fo_cases + all_lo_cases

for case in all_cases:
    print('CASE', case.name)
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.info['flops'] = case.get_flops()
    case.state = to_state_dict(case.state)
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
