"""
Product tasks

This experiment measures the ability of neural network models to generalize on
multiplicative tasks. In general, we observe linear extrapolation behavior from
ReLU models, but (ideally) perfect generalization from MNNs.
"""

# <codecell>
from dataclasses import dataclass, field
import sys
from typing import Callable

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.append('../')
from train import train
from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from model.poly import PolyConfig
from task.function import DotProductTask, MultiplicationTask

from tqdm import tqdm


def multiplication_experiment(config, domain, batch_size=128, train_iters=50_000, lr=1e-4, l1_weight=1e-4):
    task = MultiplicationTask(domain, batch_size=batch_size)
    state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=train_iters, lr=lr, l1_weight=l1_weight)
    return state, hist


def dot_product_experiment(config, domain, n_dims=5, batch_size=128, train_iters=50_000, lr=1e-4, l1_weight=1e-4, **task_args):
    task = DotProductTask(domain, n_dims=n_dims, batch_size=batch_size, **task_args)
    state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=train_iters, lr=lr, l1_weight=l1_weight)
    return state, hist


@dataclass
class Case:
    name: str
    config: dataclass
    experiment: Callable
    experiment_args: field(default_factory=dict)
    state = None
    hist = None

    def run(self):
        self.state, self.hist = self.experiment(self.config, **self.experiment_args)


# <codecell>
### DOT PRODUCT
domain = -3, 3
n_args = [2, 3]

common_args = {
    'domain': domain,
    'train_iters': 75_000
}

all_cases = []

for args in n_args:
    exp_args = dict(n_args=args, **common_args)

    all_cases.extend([
        Case('MLP', MlpConfig(n_hidden=128, n_layers=3), 
                experiment=dot_product_experiment, 
                experiment_args=exp_args),

        Case('Transformer (pure attention)', TransformerConfig(pure_linear_self_att=True), 
                experiment=dot_product_experiment, 
                experiment_args=exp_args),

        Case('Transformer (full)', TransformerConfig(use_mlp_layers=True, n_emb=None, n_hid=128, n_layers=3), 
                experiment=dot_product_experiment, 
                experiment_args=exp_args),

        Case('Mult NN', PolyConfig(n_hidden=128, n_layers=1), 
                experiment=dot_product_experiment, 
                experiment_args=exp_args)
    ])

for case in all_cases:
    print('RUNNING', case.name)
    case.run()

# <codecell>
# N_ARGS = 2 (vanilla dot-product)

curr_cases = [case for case in all_cases if case.experiment_args['n_args'] == 2]
extended_domain = -10, 10
n_pts = 50

x = np.linspace(*extended_domain, num=n_pts)
xs = np.ones((n_pts, 2, 5))
xs[:,0,0] = x
xs[:,0,1] = x

plt.plot(x, 2*x + 3, color='gray', label=r'$2x + 3$')

def plot_with(xs):
    for case in curr_cases:
        out = case.state.apply_fn({'params': case.state.params}, xs)
        plt.plot(x, out, alpha=0.8, linestyle='dashed', label=case.name, lw=2)

    plt.axvline(x=domain[0], color='red', alpha=0.2)
    plt.axvline(x=domain[1], color='red', alpha=0.2, label='train/test split')

    plt.legend()
    plt.tight_layout()

plot_with(xs)
plt.savefig('fig/dot_prod_sum.png')
plt.show()

xs = np.ones((n_pts, 2, 5))
xs[:,0,0] = x
xs[:,1,0] = x

plt.plot(x, x**2 + 4, color='black', label=r'$x^2 + 4$')
plot_with(xs)
plt.savefig('fig/dot_prod_mult.png')
plt.show()


# <codecell>
# N_ARGS = 3 ("triple" dot-product)

curr_cases = [case for case in all_cases if case.experiment_args['n_args'] == 3]
extended_domain = -10, 10
n_pts = 50

x = np.linspace(*extended_domain, num=n_pts)
xs = np.ones((n_pts, 3, 5))
xs[:,0,0] = x
xs[:,0,1] = x
xs[:,0,2] = x

plt.plot(x, 3*x + 2, color='gray', label=r'$3x + 2$')

def plot_with(xs):
    for case in curr_cases:
        out = case.state.apply_fn({'params': case.state.params}, xs)
        plt.plot(x, out, alpha=0.8, linestyle='dashed', label=case.name, lw=2)

    plt.axvline(x=domain[0], color='red', alpha=0.2)
    plt.axvline(x=domain[1], color='red', alpha=0.2, label='train/test split')

    plt.legend()
    plt.tight_layout()

plot_with(xs)
plt.savefig('fig/dot3_prod_sum.png')
plt.show()


xs = np.ones((n_pts, 3, 5))
xs[:,0,2] = x
xs[:,1,2] = x
xs[:,2,3] = x
xs[:,2,4] = x

plt.plot(x, x**2 + 2 * x + 2, color='black', label=r'$x^2 + 2x + 2$')
plot_with(xs)
plt.savefig('fig/dot3_prod_mult_quad.png')
plt.show()


xs = np.ones((n_pts, 3, 5))
xs[:,0,2] = x
xs[:,1,2] = x
xs[:,2,2] = x

plt.plot(x, x**3 + 4, color='black', label=r'$x^3 + 4$')
plot_with(xs)
plt.savefig('fig/dot3_prod_mult.png')
plt.show()


# %%
### MULTIPLICATION

domain = -3, 3
train_iters = 25_000

common_args = {
    'domain': domain,
    'train_iters': train_iters
}

all_cases = [
    Case('MLP', MlpConfig(n_hidden=32, n_layers=2), 
            experiment=multiplication_experiment, 
            experiment_args=common_args),

    Case('Mult NN', PolyConfig(n_hidden=2, n_layers=1), 
            experiment=multiplication_experiment, 
            experiment_args=common_args)
]

for case in all_cases:
    print('RUNNING', case.name)
    case.run()

# %%
extended_domain = -10, 10

x = np.linspace(*extended_domain, num=50)
xs = np.stack((x, -x), axis=-1)

for case in all_cases:
    out = case.state.apply_fn({'params': case.state.params}, xs)
    plt.plot(x, out, alpha=0.9, linestyle='dashed', label=case.name, lw=1.5)

plt.axvline(x=domain[0], color='red', alpha=0.2)
plt.axvline(x=domain[1], color='red', alpha=0.2, label='train/test split')

plt.plot(x, -x**2, color='gray', label=r'$x^2$')
plt.legend()
plt.tight_layout()
plt.savefig('fig/mult_generalization_neg.png')

# <codecell>
