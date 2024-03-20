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
from task.function import PowerTask, ClassificationTask


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
        row['train_task'].power,
        row['config']['n_layers'],
        row['config']['n_hidden'],
        row['info']['size'],
        f"{row['config']['n_layers']}-{row['config']['n_hidden']}",
        row['info']['size'] * row['train_args']['train_iters'],
        row['info']['mse'].item(),
    ], index=['name', 'train_iters', 'power', 'depth', 'width', 'size', 'arch', 'compute', 'mse'])

plot_df = df.apply(extract_plot_vals, axis=1)
plot_df

# <codecell>
### Make compute scaling laws plots

# TODO: try sigmoidal fit
def law(xs, consts):
    N, B, compute = xs
    a, b, c, d, e = consts
    # return a + b/((compute)**c)
    log_C = np.log(compute)
    return a + b/(1 + np.exp(c * log_C - d))


def get_law(law, init, resp, *feats):
    def loss(consts):
        result = law(feats, consts)
        return np.sum(huber(1e-3, (result-resp)))
        # return np.mean((result - resp)**2)

    out = minimize(loss, init)
    return out

def make_plot(name, power):
    curr_df = plot_df[(plot_df['name'] == name) & (plot_df['power'] == power)]
    plt.gcf().set_size_inches(8, 6)

    target_size = np.unique(np.sort((curr_df['size'])))[-1]

    g = sns.lineplot(curr_df, x='compute', y='mse', hue='size', marker='o')
    g.legend()
    g.legend_.set_title('size (N)')

    # t_df = curr_df[curr_df['size'] == target_size]
    # out = get_law(law, np.zeros(5), t_df['mse'], t_df['size'], t_df['train_iters'], t_df['compute'])
    # print(out)

    # xs = np.unique(t_df['compute'])
    # g.plot(xs, law((0, 0, xs), out.x), '--', color='red')
    # a, b, c, d, e = out.x
    # g.text(10**7, 1, fr'${a:.2f} + {b:.2f} \cdot C^\wedge (-{c:.3f})$', color='red')
    # g.text(10**7, 1.2, fr'${a:.2f} + {b:.2f} / (1 + (e^\wedge -{d:.2f}) \cdot C^\wedge {c:.3f})$', color='red')

    g.set_xscale('log')
    g.set_title(f'{name} (power = {power})')
    plt.tight_layout()

    return g

for p in [1,2,3]:
    make_plot('MLP', power=p)
    plt.savefig(f'fig/linreg_scale/linreg_scale_mlp_computewise_p_{p}.png')
    plt.show()

    make_plot('Transformer', power=p)
    plt.savefig(f'fig/linreg_scale/linreg_scale_transf_computewise_p_{p}.png')
    plt.show()


# <codecell>
### SCALE CLASSIFY
pkl_path = Path('remote/11_mlp_superior/scale_classify')
dfs = [pd.read_pickle(f) for f in pkl_path.iterdir() if f.suffix == '.pkl']
df = pd.concat(dfs)
df

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_args']['train_iters'],
        row['train_task'].n_classes,
        row['config']['n_layers'],
        row['config']['n_hidden'],
        row['info']['size'],
        f"{row['config']['n_layers']}-{row['config']['n_hidden']}",
        row['info']['size'] * row['train_args']['train_iters'],
        1 - row['info']['acc'].item(),
        row['info']['loss'].item(),
    ], index=['name', 'train_iters', 'n_classes', 'depth', 'width', 'size', 'arch', 'compute', 'err', 'loss'])

plot_df = df.apply(extract_plot_vals, axis=1)
plot_df

# <codecell>
### Make compute scaling laws plots

# TODO: try sigmoidal fit
def law(xs, consts):
    N, B, compute = xs
    a, b, c, d, e = consts
    # return a + b/((compute)**c)
    log_C = np.log(compute)
    return a + b/(1 + np.exp(c * log_C - d))


def get_law(law, init, resp, *feats):
    def loss(consts):
        result = law(feats, consts)
        return np.sum(huber(1e-3, (result-resp)))
        # return np.mean((result - resp)**2)

    out = minimize(loss, init)
    return out

def make_plot(name, n_classes):
    curr_df = plot_df[(plot_df['name'] == name) & (plot_df['n_classes'] == n_classes)]
    plt.gcf().set_size_inches(8, 6)

    target_size = np.unique(np.sort((curr_df['size'])))[-1]

    g = sns.lineplot(curr_df, x='compute', y='err', hue='size', marker='o')
    g.legend()
    g.legend_.set_title('size (N)')

    # t_df = curr_df[curr_df['size'] == target_size]
    # out = get_law(law, np.zeros(5), t_df['mse'], t_df['size'], t_df['train_iters'], t_df['compute'])
    # print(out)

    # xs = np.unique(t_df['compute'])
    # g.plot(xs, law((0, 0, xs), out.x), '--', color='red')
    # a, b, c, d, e = out.x
    # g.text(10**7, 1, fr'${a:.2f} + {b:.2f} \cdot C^\wedge (-{c:.3f})$', color='red')
    # g.text(10**7, 1.2, fr'${a:.2f} + {b:.2f} / (1 + (e^\wedge -{d:.2f}) \cdot C^\wedge {c:.3f})$', color='red')

    g.set_xscale('log')
    g.set_title(f'{name} (n_classes = {n_classes})')
    plt.tight_layout()

    return g

for n_classes in [2, 8, 32, 128]:
    make_plot('MLP', n_classes=n_classes)
    plt.savefig(f'fig/linreg_scale/classify_scale_mlp_computewise_nc_{n_classes}.png')
    plt.show()

    make_plot('Transformer', n_classes=n_classes)
    plt.savefig(f'fig/linreg_scale/classify_scale_transf_computewise_nc_{n_classes}.png')
    plt.show()


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
    # return 0.05 + d / (data_size**e)


def get_law(law, init, resp, *feats):
    def loss(consts):
        result = law(feats, consts)
        return np.sum(huber(1e-3, (result-resp)))
        # return np.mean((result - resp)**2)

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
# NOTE: effectively fix number of parameters, plot fit to training iterations as correct demo

# curr_df = plot_df[plot_df['name'] == 'MLP']  
curr_df = plot_df[(plot_df['name'] == 'MLP') & (plot_df['size'] == 4225)]
out = get_law(full_scaling_law,
              np.zeros(5),
              curr_df['mse'],

              curr_df['size'],
              curr_df['train_iters'])

print(out)

# <codecell>
plot_law_iterwise(curr_df, out, x=10**3, y=1, model_size_idx=-1)
plt.title('MLP')
plt.tight_layout()
# plt.savefig('fig/linreg_scale/linreg_scale_mlp_full_law_iterwise.png')
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
plot_law_sizewise(curr_df, out, x=10**3, y=0.8)
plt.title('Transformer')
plt.tight_layout()
plt.savefig('fig/linreg_scale/linreg_scale_transf_full_law_sizewise.png')

# <codecell>
### TRAINING PLAYGROUND
n_out = 1
task = ClassificationTask(n_dims=64, n_classes=n_out, seed=5, tokenize=64)
# task = PowerTask(n_dims=64, power=1, seed=5, tokenize=64)
# config = MlpConfig(n_out=n_out, n_layers=2, n_hidden=128, act_fn='relu')
config = TransformerConfig(pos_emb=True, n_out=n_out, n_layers=2, n_heads=2, n_hidden=128, n_mlp_layers=2, layer_norm=True, max_len=128)

state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=100_000, lr=1e-4)


'''
Basic observations for classification:
- Transformer handles complexity better (more classes)
- MLP handles dimensionality better (more points)
- Manually patchifying input to transformer improves it vastly (larger tokens => better performance)
    perhaps mostly because it sidesteps the attention mechanism and skips straight to MLP
    (thought its still surprisingly competent at regression with n_mlp_layers=0 / less so with classification, so not always true)
'''
# %%
