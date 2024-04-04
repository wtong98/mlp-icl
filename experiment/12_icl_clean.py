"""
Assembling the final ICL figures for NeurIPS 2024 cleanly
"""

# <codecell>
from pathlib import Path

import jax.numpy as jnp
from flax.serialization import to_state_dict
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, RfConfig, SpatialMlpConfig
from model.transformer import TransformerConfig
from task.regression import FiniteLinearRegression 

def plot_compute(df, title, hue_name='log10_size'):
    g = sns.lineplot(df, x='total_pflops', y='mse', hue=hue_name, marker='o', palette='flare_r', alpha=0.7, legend='auto')
    g.set_xscale('log')
    g.axhline(ridge_result, linestyle='dashed', color='k', alpha=0.3)

    g.set_ylabel('MSE')
    g.set_xlabel('Compute (PFLOPs)')
    g.legend_.set_title('# Params')

    for t in g.legend_.texts:
        label = t.get_text()
        t.set_text('${10}^{%s}$' % label)

    g.spines[['right', 'top']].set_visible(False)

    g.set_title(title)
    fig = g.get_figure()
    fig.tight_layout()
    return fig

fig_dir = Path('fig/final')

# <codecell>
task = FiniteLinearRegression(None, n_points=8, n_dims=8, batch_size=8192)
xs, ys = next(task)
ys_pred = estimate_ridge(None, *unpack(xs), sig2=0.05)
ridge_result = np.mean((ys_pred - ys)**2)
ridge_result

# <codecell>
# df_dir = 'remote/12_icl_clean/scale'
# pkl_path = Path(df_dir)
# dfs = [pd.read_pickle(f) for f in pkl_path.iterdir() if f.suffix == '.pkl']
# dfs = [pd.DataFrame(df[0].tolist()) for df in dfs]
# df = pd.concat(dfs)

### REGRESSION: smooth interpolation from IWL to ICL
df = collate_dfs('remote/12_icl_clean/scale')


# <codecell>
def extract_plot_vals(row):
    n_ws = row.train_task.ws
    if n_ws is not None:
        n_ws = len(n_ws)
    else:
        n_ws = float('inf')
    
    hist = row['hist']['test']
    slices = np.exp(np.linspace(-6, 0, num=10))
    slices = np.insert(slices, 0, 0)
    slice_idxs = (slices * len(hist)).astype(np.int32)
    slice_idxs[-1] -= 1  # adjust for last value
    
    hist_dict = {idx: hist[idx]['loss'].item() for idx in slice_idxs}
    print('ROW', row)

    return pd.Series([
        row['name'],
        row.train_task.n_dims,
        row.train_task.n_points,
        np.log10(row['info']['size']),
        row['info']['flops'],
        hist_dict,
        f"{row['config']['n_layers']}-{row['config']['n_hidden']}",
    ], index=['name', 'n_dims', 'n_points', 'log10_size', 'flops', 'hist', 'arch'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
#             .melt(id_vars=['name', 'n_pretrain_tasks', 'n_dims'], var_name='mse_type', value_name='mse')

def format_df(name):
    mdf = plot_df[plot_df['name'] == name].reset_index(drop=True)
    m_hist_df = pd.DataFrame(mdf['hist'].tolist())
    mdf = pd.concat((mdf[['name', 'flops', 'log10_size', 'arch']], m_hist_df), axis='columns') \
            .melt(id_vars=['name', 'flops', 'log10_size', 'arch'], var_name='hist_idx', value_name='mse')

    mdf['train_iters'] = (mdf['hist_idx'] + 1) * 1000   # 1k iterations per save 
    mdf['total_pflops'] = (mdf['flops'] * mdf['train_iters']) / 1e15
    return mdf

# <codecell>
mdf = format_df('MLP')
fig = plot_compute(mdf, 'MLP')
# fig.savefig(fig_dir / 'reg_icl_mlp_scale.svg')
fig.show()

# <codecell>
mdf = format_df('Mixer')
fig = plot_compute(mdf, 'Mixer')
# fig.savefig(fig_dir / 'reg_icl_transf_scale.svg')
fig.show()

# <codecell>
mdf = format_df('Transformer')
fig = plot_compute(mdf, 'Transformer')
# fig.savefig(fig_dir / 'reg_icl_transf_scale.svg')
fig.show()

# <codecell>
### PLOT IWL --> ICL transition
df = collate_dfs('remote/12_icl_clean/iwl_to_icl/')

def extract_plot_vals(row):
    n_ws = row.train_task.ws
    if n_ws is not None:
        n_ws = len(n_ws)
    else:
        n_ws = float('inf')
    
    return pd.Series([
        row['name'],
        n_ws,
        row.train_task.n_dims,
        row.train_task.n_points,
        row['info']['mse_pretrain'].item(),
        row['info']['mse_true'].item()
    ], index=['name', 'n_tasks', 'n_dims', 'n_points', 'mse_pretrain', 'mse_true'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
mdf = plot_df[(plot_df['name'] == 'MLP') | (plot_df['name'] == 'Mixer') | (plot_df['name'] == 'Transformer')]

ddf = plot_df[plot_df['name'] == 'dMMSE'].groupby(['n_tasks'], as_index=False).mean(['mse_pretrain', 'mse_true'])
ddf

# <codecell>
def make_iwl_to_icl_plot(mse_type):
    g = sns.lineplot(mdf, x='n_tasks', y=mse_type, hue='name', marker='o')
    g.set_xscale('log')
    g.plot(ddf['n_tasks'], ddf[mse_type], linestyle='dashed', color='purple', alpha=0.3, label='dMMSE', marker='o', markersize=5)
    g.axhline(y=ridge_result, linestyle='dashed', color='k', alpha=0.3, label='Ridge')

    g.legend()

    g.set_ylabel('MSE')
    g.set_xlabel('# Pretrain Tasks')

    g.spines[['right', 'top']].set_visible(False)

    fig = g.get_figure()
    fig.tight_layout()
    return fig

fig = make_iwl_to_icl_plot('mse_pretrain')
fig.savefig('fig/final/reg_icl_pretrain_mse.svg')
fig.clf()

fig = make_iwl_to_icl_plot('mse_true')
fig.savefig('fig/final/reg_icl_true_mse.svg')

