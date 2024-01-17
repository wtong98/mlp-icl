# <codecell>
from flax.serialization import to_state_dict
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.match import GautamMatch 
    
n_iters = 3
train_iters = 150_000
n_labels = 32

iwl_args = {'bursty': 1, 'n_classes': 128}
icl_args = {'bursty': 4, 'n_classes': 2048}
all_cases = []

for _ in range(n_iters):
    common_task_args = {'n_labels': n_labels, 'seed': new_seed()}
    common_train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'}

    iwl_tasks = [
        Case('MLP (IWL)', MlpConfig(n_out=n_labels, n_layers=3, n_hidden=512), train_args=common_train_args),
        Case('Transformer (IWL)', TransformerConfig(n_out=n_labels, n_layers=3, n_heads=4, n_hidden=512, n_mlp_layers=3), train_args=common_train_args),
        Case('MNN (IWL)', PolyConfig(n_out=n_labels, n_layers=1, n_hidden=512), train_args=common_train_args),
    ]

    for case in iwl_tasks:
        case.train_task = GautamMatch(**iwl_args, **common_task_args)
        case.test_task = GautamMatch(batch_size=1024, **iwl_args, **common_task_args)

    icl_tasks = [
        Case('MLP (ICL)', MlpConfig(n_out=n_labels, n_layers=3, n_hidden=512), train_args=common_train_args),
        Case('Transformer (ICL)', TransformerConfig(n_out=n_labels, n_layers=3, n_heads=4, n_hidden=512, n_mlp_layers=3), train_args=common_train_args),
        Case('MNN (ICL)', PolyConfig(n_out=n_labels, n_layers=1, n_hidden=512), train_args=common_train_args),
    ]

    for case in icl_tasks:
        case.train_task = GautamMatch(**icl_args, **common_task_args)
        case.test_task = GautamMatch(batch_size=1024, **icl_args, **common_task_args)

    all_cases.extend(iwl_tasks + icl_tasks)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()


# IN-DIST EVALUATION
all_tasks = [c.train_task for c in all_cases]
eval_cases(all_cases, all_tasks, key_name='acc')

# IWL EVALUATION
all_tasks = []
for case in all_cases:
    t = case.train_task
    task = GautamMatch(n_labels=t.n_labels, seed=t.seed, batch_size=1024, **iwl_args)
    task.matched_target = False
    all_tasks.append(task)

eval_cases(all_cases, all_tasks, key_name='iwl_acc')


# ICL EVALUATION (new clusters)
all_tasks = []
for case in all_cases:
    t = case.train_task
    task = GautamMatch(n_labels=t.n_labels, seed=t.seed, batch_size=1024, **icl_args)
    task.resample_clusters()
    all_tasks.append(task)

eval_cases(all_cases, all_tasks, key_name='icl_resamp_acc')


# ICL EVALUATION (swapped labels)
all_tasks = []
for case in all_cases:
    t = case.train_task
    task = GautamMatch(n_labels=t.n_labels, seed=t.seed, batch_size=1024, **icl_args)
    task.swap_labels()
    all_tasks.append(task)

eval_cases(all_cases, all_tasks, key_name='icl_swap_acc')

for case in all_cases:
    case.state = to_state_dict(case.state)

df = pd.DataFrame(all_cases)
df.to_pickle('res.pkl')

print('done!')

# %%
