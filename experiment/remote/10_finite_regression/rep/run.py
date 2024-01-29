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


run_id = new_seed()
print('RUN ID', run_id)


def t(xs):
    return np.swapaxes(xs, -2, -1)


def uninterleave(interl_xs):
    xs = interl_xs[:,0::2]
    ys = interl_xs[:,1::2,[0]]
    xs, x_q = xs[:,:-1], xs[:,[-1]]
    return xs, ys, x_q

def unpack(pack_xs):
    xs = pack_xs[:,:-1,:-1]
    ys = pack_xs[:,:-1,[-1]]
    x_q = pack_xs[:,[-1],:-1]
    return xs, ys, x_q


def estimate_dmmse(task, xs, ys, x_q, sig=0.5):
    '''
    xs: N x P x D
    ys: N x P x 1
    x_q: N x 1 x D
    ws: F x D
    '''
    ws = task.ws
    
    weights = np.exp(-(1 / (2 * sig**2)) * np.sum((ys - xs @ ws.T)**2, axis=1))  # N x F
    probs = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-32)
    w_dmmse = np.expand_dims(probs, axis=-1) * ws  # N x F x D
    w_dmmse = np.sum(w_dmmse, axis=1, keepdims=True)  # N x 1 x D
    return (x_q @ t(w_dmmse)).squeeze()


def estimate_ridge(task, xs, ys, x_q, sig=0.5):
    n_dims = xs.shape[-1]
    w_ridge = np.linalg.pinv(t(xs) @ xs + sig**2 * np.identity(n_dims)) @ t(xs) @ ys
    return (x_q @ w_ridge).squeeze()

task = FiniteLinearRegression(stack_y=False, n_ws=2, n_dims=8)
xs, ys = next(task)

y_pred = estimate_ridge(task, *uninterleave(xs))
# y_pred = estimate_dmmse(task, *uninterleave(xs))
np.mean((y_pred - ys)**2)

# <codecell>

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
train_iters = 500_000
batch_size = 128
n_dims = 4
n_ws = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, None]

# n_iters = 1
# train_iters = 1_000
# lesser_train_iters = 10
# batch_size = 256
# n_dims = 4
# n_ws = [2, None]


all_cases = []
for _ in range(n_iters):
    for n_w in n_ws:
        common_task_args = {'n_ws': n_w, 'n_dims': n_dims}

        def train_args(train_iters):
            # train_iters = 100
            return {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'}

        curr_tasks = [
            Case('MLP', MlpConfig(n_out=1, n_layers=3, n_hidden=512), train_args=train_args(train_iters)),
            # Case('MLP (id)', MlpConfig(n_out=1, n_layers=3, n_hidden=512, act_fn='linear'), train_args=common_train_args),
            Case('MLP (1-layer, relu)', MlpConfig(n_out=1, n_layers=1, n_hidden=4096), train_args=train_args(train_iters)),
            # Case('MLP (2-layer, quad)', MlpConfig(n_out=1, n_layers=2, n_hidden=2048, act_fn='quadratic'), train_args=common_train_args),
            # Case('RF (quad)', RfConfig(n_in=31*n_dims, n_hidden=2048, use_quadratic_activation=True), train_args=common_train_args),

            Case('Transformer (softmax)', TransformerConfig(pos_emb=False, n_out=1, n_layers=4, n_heads=4, n_hidden=512, n_mlp_layers=3), train_args=train_args(50_000)),
            Case('Transformer (linear)', TransformerConfig(pos_emb=False, n_out=1, n_layers=1, use_single_head_module=True, n_hidden=512, n_mlp_layers=0, layer_norm=False, softmax_att=False), train_args=train_args(100_000)),
            FunctionCase('Ridge', estimate_ridge),
        ]

        if n_w is not None:
            curr_tasks.append(FunctionCase('dMMSE', estimate_dmmse))

        for case in curr_tasks:
            seed = new_seed()
            case.train_task = FiniteLinearRegression(batch_size=batch_size, seed=seed, **common_task_args)
            case.test_task = FiniteLinearRegression(batch_size=1024, seed=seed, **common_task_args)
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
df.to_pickle(f'res.{run_id}.pkl')

print('done!')
