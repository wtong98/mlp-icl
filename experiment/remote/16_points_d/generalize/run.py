"""
Match task accuracies
"""

# <codecell>
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, RfConfig
from task.function import SameDifferentToken, SameDifferent

run_id = new_seed()
print('RUN ID', run_id)

batch_size = 128

train_iters_std = 25_000
train_iters_emb = 5_000
n_vocab = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
n_dims = [2, 4, 8, 16, 32, 64, 128, 256]

n_layers = 1
n_hidden = 512

### START TEST CONFIGS
# train_iters_std = 1000
# n_vocab = [16,]
# n_dims = [2,]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for v in n_vocab:
    for d in n_dims:
        params = {'n_vocab': v, 'n_dims': d}
        
        all_cases.extend([
            Case(f'MLP (RF)', 
                RfConfig(n_in=2*d, n_out=1, n_hidden=n_hidden, seed=new_seed()),
                train_args={'train_iters': train_iters_std, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=Finite(SameDifferent(n_dims=d, soft=False), data_size=v),
                test_task=Finite(SameDifferent(n_dims=d, soft=False), data_size=v),
                info={'params': params}),

            Case(f'MLP (fixed emb)', 
                MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden),
                train_args={'train_iters': train_iters_std, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=Finite(SameDifferent(n_dims=d, soft=False), data_size=v),
                test_task=Finite(SameDifferent(n_dims=d, soft=False), data_size=v),
                info={'params': params}),

            Case(f'MLP (learned emb)', 
                MlpConfig(n_out=1, vocab_size=2*v, n_layers=1, n_emb=2*v, n_hidden=n_hidden),
                train_args={'train_iters': train_iters_emb, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferentToken(batch_size=batch_size, n_vocab=2*v, n_seen=v),
                test_task=SameDifferentToken(batch_size=1024, n_vocab=2*v, n_seen=v),
                info={'params': params}),
        ])

        test_tasks.extend([
            SameDifferent(n_dims=d, soft=False, batch_size=1024),
            SameDifferent(n_dims=d, soft=False, batch_size=1024),
            SameDifferentToken(n_vocab=2*v, n_seen=v, sample_seen=False, batch_size=1024)
        ])


for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

seen_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=seen_tasks, key_name='acc_seen')
eval_cases(all_cases, eval_task=test_tasks, key_name='acc_unseen')

for case in all_cases:
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
