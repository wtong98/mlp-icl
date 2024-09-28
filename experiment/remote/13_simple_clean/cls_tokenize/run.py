"""
Observing scaling laws associated with model size and training iterations
"""

# <codecell>
import jax
import pandas as pd
from tqdm import tqdm

import sys


from common import *
from mlp_icl.model.transformer import TransformerConfig
from mlp_icl.task.function import ClassificationTask 

run_id = new_seed()
print('RUN ID', run_id)

batch_size = 128

train_iters_trans = 128_000
depths_trans = [4]
widths_trans = [32]

n_dims = 64
token_sizes = [1, 2, 4, 8, 16, 32, 64]
n_classes = [2, 16, 64]

### START TEST CONFIGS
# train_iters_trans = 128
# depths_trans = [4]
# widths_trans = [32]

# n_dims = 64
# token_sizes = [1]
# n_classes = [2]
### END TEST CONFIGS

all_cases = []


for depth in depths_trans:
    for width in widths_trans:
        for c in n_classes:
            for t_size in token_sizes:
                common_args = {'n_dims': n_dims, 'tokenize': t_size, 'seed': new_seed(), 'n_classes': c}

                all_cases.append(
                    Case('Transformer', TransformerConfig(n_out=c, n_layers=depth, n_hidden=width, pos_emb=True, n_mlp_layers=2),
                        train_args={'train_iters': train_iters_trans, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'},
                        train_task=ClassificationTask(batch_size=batch_size, **common_args),
                        test_task=ClassificationTask(batch_size=1024, **common_args))
                )


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
