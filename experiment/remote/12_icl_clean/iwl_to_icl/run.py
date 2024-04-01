# <codecell>
import jax.numpy as jnp
from flax.serialization import to_state_dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, RfConfig
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
        ys_pred = self.func(self.train_task, *unpack(xs), sig=task.noise_scale)
        mse = np.mean((ys - ys_pred)**2)
        self.info[key_name] = mse
    

n_iters = 1
train_iters = 1_500_000
batch_size = 128
n_points = 8
n_dims = 8
n_ws = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, None]
# NOTE: paused to do compute-optimal scaling first

# n_iters = 1
# train_iters = 1_000
# batch_size = 128
# n_points = 16
# n_dims = 8
# n_ws = [4, None]

all_cases = []
for _ in range(n_iters):
    for n_w in n_ws:
        common_task_args = {'n_ws': n_w, 'n_points': n_points, 'n_dims': n_dims, 'seed': new_seed()}
        common_train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'}

        curr_tasks = [
            Case('MLP', MlpConfig(n_out=1, n_layers=3, n_hidden=512), train_args=common_train_args),
            # TODO: add Mixer model

            Case('Transformer', TransformerConfig(n_out=1, n_layers=3, n_heads=4, n_hidden=512, n_mlp_layers=3), train_args=common_train_args),
            FunctionCase('Ridge', estimate_ridge),
        ]

        if n_w is not None:
            curr_tasks.append(FunctionCase('dMMSE', estimate_dmmse))

        for case in curr_tasks:
            case.train_task = FiniteLinearRegression(batch_size=batch_size, **common_task_args)
            case.test_task = FiniteLinearRegression(batch_size=1024, **common_task_args)
            case.info['common_task_args'] = common_task_args

        all_cases.extend(curr_tasks)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

pretrain_tasks = [c.test_task for c in all_cases]
true_tasks = []
for c in all_cases:
    task_args = c.info['common_task_args']
    task_args['n_ws'] = None
    true_tasks.append(FiniteLinearRegression(batch_size=1024, **task_args))

eval_cases(all_cases, pretrain_tasks, key_name='mse_pretrain', use_mse=True)
eval_cases(all_cases, true_tasks, key_name='mse_true', use_mse=True)

for case in all_cases:
    case.state = None

df = pd.DataFrame(all_cases)
df.to_pickle('res_mlp.pkl')

print('done!')