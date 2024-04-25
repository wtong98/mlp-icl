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

run_split = 9

n_iters = 1
train_iters_mlp = 128_000
train_iters_mix = 24_000
train_iters_transf = 16_000
batch_size = 128

n_labels = 32
n_points = 8
n_dims = 8

burstys = [1, 2, 4]
n_classes = [32, 64, 128, 256, 512, 1024, 2048]

model_depth = 8
mix_channels = 64

### START TEST CONFIGS
# run_split = 1

# n_iters = 1
# train_iters_mlp = 64_0
# train_iters_mix = 16_0
# train_iters_transf = 16_0
# batch_size = 128

# n_labels = 32
# n_points = 8
# n_dims = 8

# burstys = [1]
# n_classes = [32]

# model_depth = 1
# mix_channels = 4
### END TEST CONFIGS

all_cases = []
for _ in range(n_iters):
    for burst in burstys:
        for n_cls in n_classes:
            common_task_args = {'n_labels': n_labels, 'bursty': burst, 'n_classes': n_cls, 'n_dims': n_dims, 'seed': new_seed()}

            def make_args(train_iters):
                return {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'}

            curr_tasks = [
                Case('MLP', MlpConfig(n_out=n_labels, n_layers=model_depth, n_hidden=1024), train_args=make_args(train_iters_mlp)),
                Case('Mixer', SpatialMlpConfig(n_out=n_labels, n_layers=model_depth, n_hidden=256, n_channels=mix_channels), train_args=make_args(train_iters_mix)),
                Case('Transformer', TransformerConfig(n_out=n_labels, n_layers=model_depth, n_hidden=256, n_mlp_layers=2, pos_emb=True), train_args=make_args(train_iters_transf)),
            ]

            for case in curr_tasks:
                case.train_task = GautamMatch(batch_size=batch_size, **common_task_args)
                case.test_task = GautamMatch(batch_size=1024, **common_task_args)
                case.info['task_args'] = common_task_args

            all_cases.extend(curr_tasks)

all_cases = split_cases(all_cases, run_split)
print('ALL_CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

# IN-DIST EVALUATION
in_dist_tasks = [c.test_task for c in all_cases]
eval_cases(all_cases, in_dist_tasks, key_name='acc')

# IWL EVALUATION
all_tasks = []
for case in all_cases:
    task = GautamMatch(batch_size=1024, **case.info['task_args'])
    task.matched_target = False
    all_tasks.append(task)

eval_cases(all_cases, all_tasks, key_name='iwl_acc')


# ICL EVALUATION (new clusters)
all_tasks = []
for case in all_cases:
    task = GautamMatch(batch_size=1024, **case.info['task_args'])
    task.resample_clusters()
    all_tasks.append(task)

eval_cases(all_cases, all_tasks, key_name='icl_resamp_acc')


# ICL EVALUATION (swapped labels)
all_tasks = []
for case in all_cases:
    task = GautamMatch(batch_size=1024, **case.info['task_args'])
    task.swap_labels()
    all_tasks.append(task)

eval_cases(all_cases, all_tasks, key_name='icl_swap_acc')


for case in all_cases:
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')