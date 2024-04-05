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

run_id = new_seed()
print('RUN ID', run_id)

run_split = 8

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


n_iters = 1
train_iters_mlp = 1_024_000
train_iters_mix = 128_000
train_iters_transf = 128_000
batch_size = 128
n_points = [4, 8, 16, 32, 64, 128, 256, 512]
n_dims = [2, 4, 8, 16]
n_ws = None

model_depth = 8
mix_channels = 64

### START TEST PARAMS
# run_split = 1

# n_iters = 1
# train_iters_mlp = 1_024
# train_iters_mix = 256
# train_iters_transf = 128
# batch_size = 128
# n_points = [4]
# n_dims = [2]
# n_ws = None

# model_depth = 2
# mix_channels = 4
### END TEST PARAMS

all_cases = []
for _ in range(n_iters):
    for n_point in n_points:
        for n_dim in n_dims:
            common_task_args = {'n_ws': n_ws, 'n_dims': n_dim, 'n_points': n_point}

            def train_args(train_iters):
                return {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'}

            curr_tasks = [
                Case('MLP', MlpConfig(n_out=1, n_layers=model_depth, n_hidden=2048), train_args=train_args(train_iters_mlp)),
                Case('Mixer', SpatialMlpConfig(n_out=1, n_layers=model_depth, n_hidden=512, n_channels=mix_channels), train_args=train_args(train_iters_mix)),
                Case('Transformer', TransformerConfig(pos_emb=False, n_out=1, n_layers=model_depth, n_hidden=512, n_mlp_layers=2), train_args=train_args(train_iters_transf)),
                # FunctionCase('Ridge', estimate_ridge),
            ]

            for case in curr_tasks:
                seed = new_seed()
                case.train_task = FiniteLinearRegression(batch_size=batch_size, seed=seed, **common_task_args)
                case.test_task = FiniteLinearRegression(batch_size=8192, seed=seed, **common_task_args)
                case.info['common_task_args'] = common_task_args

            all_cases.extend(curr_tasks)

all_cases = split_cases(all_cases, run_split)
print('ALL CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

true_tasks = []
for c in all_cases:
    task_args = c.info['common_task_args']
    task_args['n_ws'] = None
    true_tasks.append(FiniteLinearRegression(batch_size=8192, **task_args))

eval_cases(all_cases, true_tasks, key_name='mse', use_mse=True)

for case in all_cases:
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')