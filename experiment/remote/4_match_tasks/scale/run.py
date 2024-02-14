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
from task.match import RingMatch 


run_id = new_seed()
print('RUN ID', run_id)

batch_size = 128

train_iters_mlp = [500, 2_000, 8_000, 32_000]
depths_mlp = [1, 2, 4, 8]
widths_mlp = [16, 128, 1024]

train_iters_trans = [500, 2_000, 8_000, 32_000]
depths_trans = [1, 2, 4, 8]
widths_trans = [16, 128, 1024]

# train_iters_mlp = [500]
# depths_mlp = [1]
# widths_mlp = [16]

# train_iters_trans = [500]
# depths_trans = [1]
# widths_trans = [16]

n_out = 6
common_args = {'n_points': n_out, 'scramble': True}

all_cases = []

for train_iters in train_iters_mlp:
    for depth in depths_mlp:
        for width in widths_mlp:
            all_cases.append(
                Case('MLP', MlpConfig(n_out=n_out, n_layers=depth, n_hidden=width),
                    train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                    train_task = RingMatch(batch_size=batch_size, **common_args),
                    test_task=RingMatch(batch_size=1024, **common_args),
                    info={'common_task_args': common_args})
            )


for train_iters in train_iters_trans:
    for depth in depths_trans:
        for width in widths_trans:
            all_cases.append(
                Case('Transformer', TransformerConfig(n_out=n_out, n_layers=depth, n_hidden=width, pos_emb=True, n_mlp_layers=3),
                    train_args={'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                    train_task = RingMatch(batch_size=batch_size, **common_args),
                    test_task=RingMatch(batch_size=1024, **common_args),
                    info={'common_task_args': common_args})
            )

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

# <codecell>
radii = [0.5, 1, 2]
for r in radii:
    eval_cases(all_cases, eval_task=RingMatch(batch_size=1024, radius=r, **common_args), key_name=f'radius_{r}')

for case in all_cases:
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

