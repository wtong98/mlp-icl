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
from task.function import PointTask

# points = [(0, 1), (-1, 0), (1, 0)]
xs = np.linspace(0, 3, num=6)
ys = (xs - 2)**2
points = np.array(list(zip(xs, ys)))

task = PointTask(points)

config = MlpConfig(n_hidden=1024, n_layers=3)
state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=10_000, lr=1e-4, l1_weight=1e-4)
# %%

xs = np.linspace(-1, 5, num=250)
out = state.apply_fn({'params': state.params}, xs)

plt.plot(xs, out)
plt.scatter(*zip(*points))
plt.axis('equal')

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
