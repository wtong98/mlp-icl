# <codecell>
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, SpatialMlpConfig
from model.transformer import TransformerConfig
from task.match import GautamMatch 

run_id = new_seed()
print('RUN ID', run_id)

run_split = 8

n_iters = 1
train_iters_mlp = 128_000
train_iters_mix = 24_000
train_iters_transf = 16_000
batch_size = 128
n_points = [4, 8, 16, 32, 64, 128, 256, 512]
n_dims = [2, 4, 8, 16]

n_classes = 2048
n_labels = 32

model_depth = 8
mix_channels = 64

### START TEST PARAMS
# run_split = 1

# n_iters = 1
# train_iters_mlp = 64_0
# train_iters_mix = 16_0
# train_iters_transf = 16_0
# batch_size = 128
# n_points = [4]
# n_dims = [2]

# n_classes = 2048
# n_labels = 32

# model_depth = 1
# mix_channels = 4
### END TEST PARAMS

all_cases = []
for _ in range(n_iters):
    for n_point in n_points:
        for n_dim in n_dims:
            common_task_args = {'n_labels': n_labels, 'bursty': n_point // 2, 'n_classes': n_classes, 'n_dims': n_dim, 'n_points': n_point, 'seed': new_seed()}

            def train_args(train_iters):
                return {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'}

            curr_tasks = [
                Case('MLP', MlpConfig(n_out=n_labels, n_layers=model_depth, n_hidden=1024), train_args=train_args(train_iters_mlp)),
                Case('Mixer', SpatialMlpConfig(n_out=n_labels, n_layers=model_depth, n_hidden=256, n_channels=mix_channels), train_args=train_args(train_iters_mix)),
                Case('Transformer', TransformerConfig(pos_emb=True, n_out=n_labels, n_layers=model_depth, n_hidden=256, n_mlp_layers=2, max_len=2048), train_args=train_args(train_iters_transf)),
            ]

            for case in curr_tasks:
                case.train_task = GautamMatch(batch_size=batch_size, **common_task_args)
                case.test_task = GautamMatch(batch_size=1024, **common_task_args)
                case.info['common_task_args'] = common_task_args

            all_cases.extend(curr_tasks)

all_cases = split_cases(all_cases, run_split)
print('ALL CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

for case in all_cases:
    case.state = None
    case.info['loss'] = case.hist['test'][-1].loss

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')