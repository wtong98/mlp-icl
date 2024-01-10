"""
Experimenting with points, on a whim
"""

# <codecell>
import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from common import *

import sys
sys.path.append('../')
from train import train
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from model.knn import KnnConfig
from task.function import PointTask

# points = [(0, 1), (-1, 0), (1, 0)]
n_iters = 5

n_points = [2, 3, 4, 5]
all_states = []
all_points = []

for pts in n_points:
  states = []

  xs = np.linspace(1, 5, num=pts)
  ys = (xs - 3)**2
  points = np.array(list(zip(xs, ys)))
  all_points.append(points)
  task = PointTask(points)

  for _ in range(n_iters):
    config = MlpConfig(n_hidden=128, n_layers=3)
    state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=10_000, lr=1e-4, l1_weight=1e-4)
    states.append(state)

  all_states.append(states)

all_states = np.array(all_states)
# %%
import jax.numpy as jnp

def compute_dists(point, data):
    '''
    point: B x D
    data:  N x D
    '''
    diff = data - jnp.expand_dims(point, axis=1)
    dot = jnp.einsum('bnd,bmd->bnm', diff, diff)

    dists = jnp.sqrt(jnp.diagonal(dot, axis1=1, axis2=2))
    return dists


@dataclass
class KnnConfig:
    """Global hyperparamters"""
    beta: float = 1
    n_classes: int | None = None
    xs: np.ndarray = None
    ys: np.ndarray = None

    def to_model(self):
        return Knn(self)


class Knn:
    def __init__(self, config) -> None:
        self.config = config
    
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)

        dists = compute_dists(x, self.config.xs)
        weights = jnp.exp(-self.config.beta * dists)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        if self.config.n_classes is None:
            return jnp.sum(weights * self.config.ys, axis=1)
        else:
            props = jnp.stack([jnp.sum(weights[:,y==self.config.ys], axis=1) for y in np.arange(self.config.n_classes)], axis=1)
            return props


fig, axs = plt.subplots(1, 4, figsize=(10, 2))

for ax, pts, states in zip(axs.ravel(), all_points, all_states):
  xs = np.linspace(-1, 7, num=250)

  for i, state in enumerate(states):
    out = state.apply_fn({'params': state.params}, xs)

    if i == 0:
      ax.plot(xs, out, color='C0', alpha=0.3, label='MLP')
    else:
      ax.plot(xs, out, color='C0', alpha=0.3)

  ax.scatter(*zip(*pts), label='data')
  ax.axis('equal')

  knn = KnnConfig(beta=1.5, xs=pts[:,[0]], ys=pts[:,1]).to_model()
  knn_preds = knn(xs)
  ax.plot(xs, knn_preds, color='C1', alpha=0.7, label='near-neigh')

plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('fig/points_interpolation_with_nn.png')

'''
Roughy summary:
- within interpolation region, MLP mostly draws straight lines
between points
- weirdness about the origin -- tends to form cusps
- will favor cusp over symmetry about the origin

- **MLP can be caricatured by nearest-neighbor approach**
- key question of inductive bias: how to design *model geometry* that
  aligns with *task geometry*?
- philosophical point: can't distinguish what is perfect mimicry
  from the real McCoy
'''
