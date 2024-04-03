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
from task.regression import FiniteLinearRegression 

@dataclass
class FunctionCase:
    name: str
    func: Callable
    train_task: Iterable | None = None
    test_task: Iterable | None = None
    info: dict = field(default_factory=dict)

    def run(self):
        return
    
    def eval_mse(self, task, key_name='eval_mse'):
        xs, ys = next(task)
        ys_pred = self.func(self.train_task, *unpack(xs), sig2=task.noise_scale)
        mse = np.mean((ys - ys_pred)**2)
        self.info[key_name] = mse
    
run_split = 7

# TODO: refine train_iters to be FLOP scaled
n_iters = 1
train_iters_mlp = 1_024_000
train_iters_mix = 256_000
train_iters_transf = 128_000
batch_size = 128
n_points = 8
n_dims = 8
n_ws = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, None]

model_depth = 8
mix_channels = 128

### START TEST CONFIGS
# run_split = 1
# train_iters_mlp = 1_024
# train_iters_mix = 256
# train_iters_transf = 128
# batch_size = 128
# n_points = 8
# n_dims = 8
# n_ws = [2]

# model_depth = 1
# mix_channels = 4
### END TEST CONFIGS

all_cases = []
for _ in range(n_iters):
    for n_w in n_ws:
        common_task_args = {'n_ws': n_w, 'n_points': n_points, 'n_dims': n_dims, 'seed': new_seed()}

        def make_args(train_iters):
            return {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'}

        curr_tasks = [
            Case('MLP', MlpConfig(n_out=1, n_layers=model_depth, n_hidden=2048), train_args=make_args(train_iters_mlp)),
            Case('Mixer', SpatialMlpConfig(n_out=1, n_layers=model_depth, n_hidden=1024, n_channels=mix_channels), train_args=make_args(train_iters_mlp)),
            Case('Transformer', TransformerConfig(n_out=1, n_layers=model_depth, n_hidden=512, n_mlp_layers=2, pos_emb=False), train_args=make_args(train_iters_transf)),
            FunctionCase('Ridge', estimate_ridge),
        ]

        if n_w is not None:
            curr_tasks.append(FunctionCase('dMMSE', estimate_dmmse))

        for case in curr_tasks:
            case.train_task = FiniteLinearRegression(batch_size=batch_size, **common_task_args)
            case.test_task = FiniteLinearRegression(batch_size=8192, **common_task_args)
            case.info['common_task_args'] = common_task_args

        all_cases.extend(curr_tasks)

all_cases = split_cases(all_cases, run_split)
print('ALL_CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

pretrain_tasks = [c.test_task for c in all_cases]
true_tasks = []
for c in all_cases:
    task_args = c.info['common_task_args']
    task_args['n_ws'] = None
    true_tasks.append(FiniteLinearRegression(batch_size=8192, **task_args))

eval_cases(all_cases, pretrain_tasks, key_name='mse_pretrain', use_mse=True)
eval_cases(all_cases, true_tasks, key_name='mse_true', use_mse=True)

for case in all_cases:
    case.state = None
    case.hist = None

df = pd.DataFrame(all_cases)
df.to_pickle('res.pkl')

print('done!')