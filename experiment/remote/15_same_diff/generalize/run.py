"""
Match task accuracies
"""

# <codecell>
import jax
import pandas as pd
from tqdm import tqdm

import sys


from common import *
from mlp_icl.model.mlp import MlpConfig
from mlp_icl.task.function import SameDifferentToken

run_id = new_seed()
print('RUN ID', run_id)

batch_size = 128

train_iters_mlp = 10_000
n_vocab = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

### START TEST CONFIGS
# train_iters_mlp = 1000
# n_vocab = [8]
### END TEST CONFIGS

all_cases = []

for v in n_vocab:
    common_args = {'n_vocab': v, 'n_seen': v // 2}

    all_cases.extend([
        Case('MLP', MlpConfig(n_out=1, vocab_size=v, n_layers=4, n_emb=256, n_hidden=256),
            train_args={'train_iters': train_iters_mlp, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
            train_task=SameDifferentToken(batch_size=batch_size, **common_args),
            test_task=SameDifferentToken(batch_size=1024, **common_args),
            info={'common_args': common_args}),
    ])


for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=tasks, key_name='acc_seen')

for t in tasks:
    t.sample_seen = False

eval_cases(all_cases, eval_task=tasks, key_name='acc_unseen')

for case in all_cases:
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
