"""
Match task accuracies
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
from task.match import RingMatch

run_id = new_seed()
print('RUN ID', run_id)

batch_size = 128

train_iters_dot = 128_000

train_iters_mlp = 8_000
depth_mlp = 4
width_mlp = 256

train_iters_trans = 8_000
depth_trans = 4
width_trans = 32

n_points = 6
do_scramble = [False, True]
radii = [0.25, 0.5, 1, 2, 4, 8, 16]

### START TEST CONFIGS
# train_iters_dot = 128

# train_iters_mlp = 8_0
# depth_mlp = 4
# width_mlp = 256

# train_iters_trans = 8_0
# depth_trans = 4
# width_trans = 32

# n_points = 6
# do_scramble = [False]
# radii = [1]
### END TEST CONFIGS

all_cases = []

for scram in do_scramble:
    for r in radii:
        common_args = {'n_points': n_points, 'scramble': scram, 'radius': 1}

        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_points, n_layers=depth_mlp, n_hidden=width_mlp),
                train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = RingMatch(batch_size=batch_size, **common_args),
                test_task=RingMatch(batch_size=1024, **common_args),
                info={'common_args': common_args}),

            Case('Transformer', TransformerConfig(n_out=n_points, n_layers=depth_trans, n_hidden=width_trans, pos_emb=True, n_mlp_layers=2),
                train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = RingMatch(batch_size=batch_size, **common_args),
                test_task=RingMatch(batch_size=1024, **common_args),
                info={'common_args': common_args}),

            Case('RB MLP', 
                DotMlpConfig(n_out=n_points, use_initial_proj=False, last_token_only=True),
                train_args={'train_iters': train_iters_dot, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = RingMatch(batch_size=batch_size, **common_args),
                test_task=RingMatch(batch_size=1024, **common_args),
                info={'common_args': common_args})
        ])


for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

for scram in do_scramble:
    for r in radii:
        test_task = RingMatch(batch_size=1024, n_points=n_points, scramble=scram, radius=r)
        eval_cases(all_cases, eval_task = test_task, key_name=f'acc_{scram}_{r}')

for case in all_cases:
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.info['flops'] = case.get_flops()
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
