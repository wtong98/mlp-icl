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
from task.function import SameDifferent

run_id = new_seed()
print('RUN ID', run_id)

run_split = 3

train_iters_std = 25_000
train_iters_emb = 5_000
n_vocab = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_dims = [2, 4, 8, 16, 32, 64, 128]
gs = [0.01, 1, 100]
base_lr = 0.1
print(n_dims)

n_layers = 1
n_hidden = 256

### START TEST CONFIGS
# run_split = 1

# train_iters_std = 25_00
# train_iters_emb = 5_00
# n_vocab = [4]
# n_dims = [2]
# gs = [100]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for v in n_vocab:
    for d in n_dims:
        params = {'n_symbols': v, 'n_dims': d}
        
        all_cases.extend([
            Case(f'MLP (RF)', 
                RfConfig(n_in=2*d, n_out=1, n_hidden=n_hidden, seed=new_seed()),
                train_args={'train_iters': train_iters_std, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferent(n_symbols=v, n_dims=d),
                test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024)),

            Case(f'MLP (fixed emb)', 
                MlpConfig(n_out=1, n_layers=1, n_hidden=n_hidden),
                train_args={'train_iters': train_iters_std, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferent(n_symbols=v, n_dims=d),
                test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024)),

            Case(f'MLP (learned emb)', 
                MlpConfig(n_out=1, vocab_size=2*v, n_layers=1, n_emb=2*v, n_hidden=n_hidden),
                train_args={'train_iters': train_iters_emb, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce'},
                train_task=SameDifferent(task='token', n_symbols=2*v, n_dims=d),
                test_task=SameDifferent(task='token', n_symbols=2*v, sample_seen=False, batch_size=1024)),
        ])

        for g in gs:
            all_cases.append(
                Case(f'MLP (SGD, g={g})',
                     MlpConfig(mup_scale=True, n_out=1, n_layers=1, n_hidden=n_hidden, feature_learning_strength=g),
                     train_args={'train_iters': train_iters_std, 'test_iters': 1, 'test_every': 1000, 'loss': 'bce',
                                 'optim': optax.sgd, 'lr': g * base_lr},
                     train_task=SameDifferent(n_symbols=v, n_dims=d),
                     test_task=SameDifferent(n_symbols=None, n_dims=d, batch_size=1024))
            )


all_cases = split_cases(all_cases, run_split)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

train_tasks = [c.train_task for c in all_cases]
test_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, eval_task=train_tasks, key_name='acc_seen')
eval_cases(all_cases, eval_task=test_tasks, key_name='acc_unseen')

for case in all_cases:
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
