# <codecell>
import numpy as np
import pandas as pd
from tqdm import tqdm


import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.oddball import FreeOddballTask 
    

n_iters = 5
n_out = 6
data_sizes = [8, 16, 32, 64, 128, 256, 512]
train_iters=30_000

all_cases = []
for _ in range(n_iters):
    for size in data_sizes:

        common_seed = new_seed()
        common_task_args = {'n_choices': n_out, 'seed': common_seed}

        common_train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000}
        early_stop_train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 25, 
                             'early_stop_n': 3, 'early_stop_key': 'accuracy', 'early_stop_decision': 'max'}

        curr_tasks = [
            Case('MLP (full)', MlpConfig(n_out=n_out, n_layers=3, n_hidden=256), train_args=common_train_args),
            Case('MLP (1 layer)', MlpConfig(n_out=n_out, n_layers=1, n_hidden=256), train_args=common_train_args),
            Case('MLP (early stop)', MlpConfig(n_out=n_out, n_layers=1, n_hidden=256), train_args=early_stop_train_args),

            Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=256, use_mlp_layers=True, pos_emb=True), train_args=common_train_args),
            Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=256), train_args=common_train_args),
            KnnCase('KNN', KnnConfig(beta=3, n_classes=n_out), task_class=FreeOddballTask, data_size=size, seed=common_seed)
        ]

        for case in curr_tasks:
            case.train_task = FreeOddballTask(data_size=size, **common_task_args)
            case.test_task = FreeOddballTask(batch_size=1024, **common_task_args)

        all_cases.extend(curr_tasks)


for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

task = FreeOddballTask(n_choices=n_out, batch_size=1024)
eval_cases(all_cases, task)

# quick-fix purge
for case in all_cases:
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle('res.pkl')

print('done!')