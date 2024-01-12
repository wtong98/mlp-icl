"""
Can these models learn regression in-context? We're about to find out!

Observations:
- MLP achieves relatively low MSE for larger x ranges, all models about the same
for larger w's (with transformer perhaps being slightly better)
- Reading cross-sections is a bit like reading tea leaves. The models tend to
track the line closely around the interpolation region, but quickly fall off
outside

Things to try:
- larger dimensionality
- comparison to KNN

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import jax
from flax.serialization import from_state_dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from common import *

import sys
sys.path.append('../')
from train import train, create_train_state
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegression


# <codecell>
df = pd.read_pickle('remote/8_regression_tasks/res.pkl')

# <codecell>
def extract_plot_vals(row, argname):
    args = {key : val.item() for key, val in row['info'].items() if argname in key}
    xs = [key.split('_')[-1] for key in args]
    xs = [int(key) for key in xs]
    vals = list(args.values())

    return pd.Series([row['name']] + vals, 
                     index=['name'] + xs)

plot_df_xlim = df.apply(lambda row: extract_plot_vals(row, 'x_lim'), axis=1) \
                 .melt(id_vars=['name'],
                       var_name='x_lim',
                       value_name='mse')


plt.gcf().set_size_inches(6, 3)
g = sns.barplot(plot_df_xlim, x='x_lim', y='mse', hue='name')
g.legend_.set_title(None)
g.set_yscale('log')

plt.tight_layout()
plt.savefig('fig/reg_xlim_gen.png')
plt.show()


# <codecell>
plot_df_wscale = df.apply(lambda row: extract_plot_vals(row, 'w_scale'), axis=1) \
                   .melt(id_vars=['name'],
                         var_name='w_scale',
                         value_name='mse')

plt.gcf().set_size_inches(6, 3)
g = sns.barplot(plot_df_wscale, x='w_scale', y='mse', hue='name')
g.legend_.set_title(None)
g.set_yscale('log')

plt.tight_layout()
plt.savefig('fig/reg_wscale_gen.png')
plt.show()


# <codecell>
def load_state(idx, config_class):
      c = df.iloc[idx]
      model = config_class(**c.config).to_model()
      task = LinearRegression()
      xs, _ = next(task)

      target = create_train_state(
            jax.random.PRNGKey(new_seed()),
            model,
            xs)

      state = from_state_dict(target, c.state)
      return state

# <codecell>
cases = [
     (0, MlpConfig),
     (1, TransformerConfig),
     (2, PolyConfig)
]

def plot_extrap(w, save_path):
      num_points = 100

      xs = np.random.uniform(-5, 5, size=5)
      ys = xs * w

      test_x = np.linspace(-100, 100, num=num_points)
      test_y = w * test_x

      plt.plot(test_x, test_y, '--', color='gray', alpha=0.5, label='true')

      for c in cases:
            state = load_state(*c)

            points = np.zeros((num_points, 11, 2))
            points[:,0:-1:2,0] = xs
            points[:,1:-1:2,0] = ys
            points[np.arange(num_points),-1,0] = test_x

            y_pred = state.apply_fn({'params': state.params}, points)
            plt.plot(test_x, y_pred, label=df.iloc[c[0]]['name'], alpha=0.8)

      plt.axvline(x=-5, color='red', alpha=0.1)
      plt.axvline(x=5, color='red', alpha=0.1)

      plt.legend()
      plt.tight_layout()
      plt.savefig(save_path)
      plt.show()

plot_extrap(w=0, save_path='fig/reg_extrap_plot_w_0.png')
plot_extrap(w=1, save_path='fig/reg_extrap_plot_w_1.png')
plot_extrap(w=5, save_path='fig/reg_extrap_plot_w_5.png')


# <codecell>
task = LinearRegression(batch_size=128, n_dims=5)
# config = MlpConfig(n_out=1, n_layers=3, n_hidden=512)
# config = PolyConfig(n_out=1, n_layers=1, n_hidden=512, start_with_dense=True)
config = TransformerConfig(pos_emb=True, n_out=1, n_heads=4, n_layers=3, n_hidden=512, use_mlp_layers=False)

state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=100_000, lr=1e-4, l1_weight=1e-4)


# <codecell>
xs = np.array([[1, 0], [1, 0], 
               [0, 1], [1, 0],
               [1, 1], [1, 0],
               [-1, -1], [-2, 0],
               [-1, 0], [-1, 0],
               [5, -1]])

xs = np.expand_dims(xs, axis=0)
state.apply_fn({'params': state.params}, xs)

# <codecell>

from flax.serialization import from_state_dict, to_state_dict

from_state_dict(state, to_state_dict(state)).apply_fn
