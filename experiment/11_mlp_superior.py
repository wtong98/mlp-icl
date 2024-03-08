"""
Searching for tasks where MLP > transformer

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import huber
import pandas as pd
import seaborn as sns

from common import *

import sys
sys.path.append('../')
from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from task.function import LinearTask


# <codecell>
### PLOTTING SCALING
pkl_path = Path('remote/11_mlp_superior/scale')
dfs = [pd.read_pickle(f) for f in pkl_path.iterdir() if f.suffix == '.pkl']
df = pd.concat(dfs)
df


def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_args']['train_iters'],
        row['config']['n_layers'],
        row['config']['n_hidden'],
        row['info']['size'],
        f"{row['config']['n_layers']}-{row['config']['n_hidden']}",
        row['info']['mse'].item(),
    ], index=['name', 'train_iters', 'depth', 'width', 'size', 'arch', 'mse'])

plot_df = df.apply(extract_plot_vals, axis=1)
plot_df

# <codecell>
def plot_panel(name, feat, hue, save_path):
    curr_df = plot_df[(plot_df['name'] == name)]
    g = sns.lineplot(curr_df, x=feat, y='mse', hue=hue, markers=True, marker='o')

    g.set_yscale('log')
    g.set_xscale('log')
    g.set_title(name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

plot_panel('MLP', 'size', 'train_iters', 'fig/linreg_scale/linreg_scale_mlp_sizewise.png')
plot_panel('MLP', 'train_iters', 'size', 'fig/linreg_scale/linreg_scale_mlp_iterwise.png')
plot_panel('MLP', 'train_iters', 'arch', 'fig/linreg_scale/linreg_scale_mlp_archwise.png')

plot_panel('Transformer', 'size', 'train_iters', 'fig/linreg_scale/linreg_scale_transf_sizewise.png')
plot_panel('Transformer', 'train_iters', 'size', 'fig/linreg_scale/linreg_scale_transf_iterwise.png')
plot_panel('Transformer', 'train_iters', 'arch', 'fig/linreg_scale/linreg_scale_transf_archwise.png')

# <codecell>
### PLOT FULL SCALING LAWS
def full_scaling_law(xs, consts):
    model_size, data_size = xs
    a, b, c, d, e = consts
    return a + b / (model_size**c) + d / (data_size**e)


def get_law(law, init, resp, *feats):
    def loss(consts):
        result = law(feats, consts)
        return np.sum(huber(1e-3, (result-resp)))

    out = minimize(loss, init)
    return out


def plot_law_sizewise(df, out, model_idx=-1, x=0, y=0):
    g = sns.lineplot(data=df, x='size', y='mse', hue='train_iters', markers=True, marker='o')
    g.set_xscale('log')

    target = np.sort(np.unique(df['train_iters']))[model_idx]
    target_df = df[df['train_iters'] == target]
    xs =  np.sort(target_df['size'])
    g.plot(xs, full_scaling_law((xs, target), out.x), '--', color='red')
    a, b, c, d, e = out.x
    g.text(x, y, fr'${a:.2f} + {b:.2f} N^\wedge (-{c:.3f}) + {d:.2f} B^\wedge (-{e:.3f})$', color='red')


def plot_law_iterwise(df, out, model_size_idx=-1, x=0, y=0):
    g = sns.lineplot(data=df, x='train_iters', y='mse', hue='size', markers=True, marker='o')
    g.set_xscale('log')

    model_size = np.sort(np.unique(df['size']))[model_size_idx]
    target_df = df[df['size'] == model_size]
    xs =  np.sort(target_df['train_iters'])
    g.plot(xs, full_scaling_law((model_size, xs), out.x), '--', color='red')
    a, b, c, d, e = out.x
    g.text(x, y, fr'${a:.2f} + {b:.2f} N^\wedge (-{c:.3f}) + {d:.2f} B^\wedge (-{e:.3f})$', color='red')

# <codecell>
curr_df = plot_df[plot_df['name'] == 'MLP']
out = get_law(full_scaling_law,
              np.zeros(5),
              curr_df['mse'],

              curr_df['size'],
              curr_df['train_iters'])

plot_law_iterwise(curr_df, out, x=10**3, y=1)
plt.title('MLP')
plt.tight_layout()
plt.savefig('fig/linreg_scale/linreg_scale_mlp_full_law_iterwise.png')
plt.show()

# <codecell>
plot_law_sizewise(curr_df, out, x=10**3, y=1)
plt.title('MLP')
plt.tight_layout()
plt.savefig('fig/linreg_scale/linreg_scale_mlp_full_law_sizewise.png')
plt.show()

# <codecell>
curr_df = plot_df[plot_df['name'] == 'Transformer']
out = get_law(full_scaling_law,
              np.zeros(5),
              curr_df['mse'],

              curr_df['size'],
              curr_df['train_iters'])

print(out)

# <codecell>
plot_law_iterwise(curr_df, out, x=10**3, y=1)
plt.title('Transformer')
plt.tight_layout()
plt.savefig('fig/linreg_scale/linreg_scale_transf_full_law_iterwise.png')

# <codecell>
plot_law_sizewise(curr_df, out, x=10**5, y=0.8)
plt.title('Transformer')
plt.tight_layout()
plt.savefig('fig/linreg_scale/linreg_scale_transf_full_law_sizewise.png')

# <codecell>
### TRAINING PLAYGROUND

task = LinearTask(n_dims=64, seed=5, tokenize=True)
# config = MlpConfig(n_out=1, n_layers=3, n_hidden=128, act_fn='relu')
config = TransformerConfig(pos_emb=True, n_out=1, n_layers=3, n_heads=2, n_hidden=256, n_mlp_layers=3, layer_norm=True, max_len=128)

state, hist = train(config, data_iter=iter(task), loss='mse', test_every=100, train_iters=5000, lr=1e-4)
# %%
