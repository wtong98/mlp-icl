"""
Observing scaling laws associated with model size and training iterations
"""

# <codecell>
import jax
import pandas as pd
from tqdm import tqdm

import sys


from common import *
from mlp_icl.model.mlp import MlpConfig , SpatialMlpConfig
from mlp_icl.model.transformer import TransformerConfig
from mlp_icl.task.match import GautamMatch 

run_id = new_seed()
print('RUN ID', run_id)


batch_size = 128

run_split = 3

train_iters_mlp = 128_000
depths_mlp = [2, 4, 8]
widths_mlp = [64, 256, 1024]

train_iters_mix = 24_000
depths_mix = [2, 4, 8]
widths_mix = [16, 64, 256]
channels_mix = 64

train_iters_trans = 16_000
depths_trans = [2, 4, 8]
widths_trans = [16, 64, 256]

n_labels = 32
n_dims = 8
n_points = 8
n_classes = None
bursty = 4

### START TEST CONFIGS
# run_split = 1

# train_iters_mlp = 64_0
# depths_mlp = [2]
# widths_mlp = [64]

# train_iters_mix = 16
# depths_mix = [2]
# widths_mix = [16]
# channels_mix = 64

# train_iters_trans = 16
# depths_trans = [2]
# widths_trans = [16]
### END TEST CONFIGS


all_cases = []

for depth in depths_mlp:
    for width in widths_mlp:
        common_task_args = {'n_labels': n_labels, 'bursty': bursty, 'n_dims': n_dims, 'n_classes': n_classes, 'seed': new_seed()}

        all_cases.append(
            Case('MLP', MlpConfig(n_out=n_labels, n_layers=depth, n_hidden=width),
                train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = GautamMatch(batch_size=batch_size, **common_task_args),
                test_task=GautamMatch(batch_size=1024, **common_task_args),
                info={'common_task_args': common_task_args})
        )

for depth in depths_mix:
    for width in widths_mix:
        common_task_args = {'bursty': bursty, 'n_dims': n_dims, 'n_classes': n_classes, 'seed': new_seed()}

        all_cases.append(
            Case('Mixer', SpatialMlpConfig(n_out=n_labels, n_layers=depth, n_hidden=width, n_channels=channels_mix),
                train_args={'train_iters': train_iters_mix, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = GautamMatch(batch_size=batch_size, **common_task_args),
                test_task=GautamMatch(batch_size=1024, **common_task_args),
                info={'common_task_args': common_task_args})
        )


for depth in depths_trans:
    for width in widths_trans:
        common_task_args = {'bursty': bursty, 'n_dims': n_dims, 'n_classes': n_classes, 'seed': new_seed()}

        all_cases.append(
            Case('Transformer', TransformerConfig(n_out=n_labels, n_layers=depth, n_hidden=width, pos_emb=True, n_heads=1, n_mlp_layers=2),
                train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                train_task = GautamMatch(batch_size=batch_size, **common_task_args),
                test_task=GautamMatch(batch_size=1024, **common_task_args),
                info={'common_task_args': common_task_args})
        )

all_cases = split_cases(all_cases, run_split)
print('CURRENT CASES:', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

for case in all_cases:
    case.info['size'] = sum(x.size for x in jax.tree_util.tree_leaves(case.state.params))
    case.info['flops'] = case.get_flops()
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
