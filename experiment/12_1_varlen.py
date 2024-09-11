"""
Assembling the final ICL figures for NeurIPS 2024 cleanly
"""

# <codecell>
from pathlib import Path

import jax.numpy as jnp
from flax.serialization import to_state_dict
import matplotlib.pyplot as plt
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

set_theme()

def plot_compute(df, title, hue_name='log10_size'):
    g = sns.lineplot(df, x='total_pflops', y='mse', hue=hue_name, marker='o', palette='flare_r', alpha=0.7, legend='auto')
    g.set_xscale('log')
    g.axhline(ridge_result, linestyle='dashed', color='r', alpha=0.5)

    g.set_ylabel('MSE')
    g.set_xlabel('Compute (PFLOPs)')
    g.legend_.set_title('# Params')

    for t in g.legend_.texts:
        label = t.get_text()
        t.set_text('${10}^{%s}$' % label)

    g.set_title(title)
    fig = g.get_figure()
    fig.tight_layout()
    return fig

fig_dir = Path('fig/final')

# <codecell>
all_ridge_results = []
for n in range(3, 8):
    task = FiniteLinearRegression(None, n_points=n, n_dims=8, batch_size=8192)
    xs, ys = next(task)
    ys_pred = estimate_ridge(None, *unpack(xs), sig2=0.05)
    ridge_result = np.mean((ys_pred - ys)**2)
    all_ridge_results.append(ridge_result)

ridge_result = np.mean(all_ridge_results)
ridge_result

# <codecell>
### REGRESSION: smooth interpolation from IWL to ICL
df = collate_dfs('remote/12_icl_clean/varlen_scale')
df

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

def format_df(name=None):
    mdf = plot_df.copy()
    if name is not None:
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
fig.savefig(fig_dir / 'varlen_mlp_scale.svg')
fig.show()

# <codecell>
mdf = format_df('Mixer')
fig = plot_compute(mdf, 'Mixer')
fig.savefig(fig_dir / 'varlen_mix_scale.svg')
fig.show()

# <codecell>
mdf = format_df('Transformer')
fig = plot_compute(mdf, 'Transformer')
fig.savefig(fig_dir / 'varlen_transf_scale.svg')
fig.show()

# <codecell>
mdfs = [format_df('MLP'), format_df('Mixer'), format_df('Transformer')]
# mdfs = [format_df('Transformer'), format_df('Mixer'), format_df('MLP')]
mdf = pd.concat(mdfs)
# mdf = pd.concat(mdfs).sample(frac=1)
# mdf = mdf[::4].sample(frac=1)
mdf = mdf.sample(frac=1)[::5]

g = sns.scatterplot(mdf, x='total_pflops', y='mse', hue='name', marker='o', alpha=0.7, legend='auto', s=50, hue_order=['MLP', 'Mixer', 'Transformer'])
g.set_xscale('log')
g.axhline(ridge_result, linestyle='dashed', color='r', alpha=0.5)

g.legend_.set_title(None)

g.set_ylabel('MSE')
g.set_xlabel('Compute (PFLOPs)')
g.set_title('ICL Regression')

fig = g.get_figure()
fig.set_size_inches(4, 3)
fig.tight_layout()
# fig.savefig(fig_dir / 'varlen_reg_icl_all_scale.svg')
fig.savefig(fig_dir / 'varlen_reg_icl_all_scale.png')

# <codecell>
### PLOT IWL --> ICL transition
df = collate_dfs('remote/12_icl_clean/varlen_iwl_to_icl/')

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

plot_df = plot_df[plot_df['n_tasks'] != 2]
plot_df = plot_df[plot_df['n_tasks'] != float('inf')]

mdf = plot_df[(plot_df['name'] == 'MLP') | (plot_df['name'] == 'Mixer') | (plot_df['name'] == 'Transformer')]
mdf[(mdf['name'] == 'Transformer') & (mdf['mse_pretrain'] > 0.65)] = None   # remove failing seeds
mdf = mdf.dropna()
mdf

# <codecell>
# construct ridge estimates
ks = np.unique(mdf['n_tasks'])
ridge_res = []

for _ in tqdm(range(100)):
    for k in ks:
        k = int(k)

        all_dmmse = []
        all_dmmse_unr = []
        all_ridge = []
        for n in range(3, 8+1):
            task = FiniteLinearRegression(k, n_points=n, n_dims=8, batch_size=128)
            xs, ys = next(task)
            ys_pred_dm = estimate_dmmse(task, *unpack(xs), sig2=0.05)
            ys_pred = estimate_ridge(None, *unpack(xs), sig2=0.05)

            dm_result = np.mean((ys_pred_dm - ys)**2)
            all_dmmse.append(dm_result)
            ridge_result = np.mean((ys_pred - ys)**2)
            all_ridge.append(ridge_result)

            task_unr = FiniteLinearRegression(None, n_points=n, n_dims=8, batch_size=128)
            xs, ys = next(task_unr)
            ys_pred_dm_unr = estimate_dmmse(task, *unpack(xs), sig2=0.05)
            all_dmmse_unr.append(np.mean((ys_pred_dm_unr - ys)**2))
        
        ridge_res.append({
            'n_tasks': k,
            'ridge_mse': np.mean(all_ridge),
            'dmmse_mse': np.mean(all_dmmse),
            'dmmse_mse_unr': np.mean(all_dmmse_unr)
        })

rdf = pd.DataFrame(ridge_res)
rdf

# <codecell>
def make_iwl_to_icl_plot(mse_type, title='', ylim=False, sim=False):
    g = sns.lineplot(mdf, x='n_tasks', y=mse_type, hue='name', marker='o', alpha=0.9, estimator='mean', markersize=8)
    g.set_xscale('log', base=2)
    if ylim:
        g.set_ylim(0, 0.55)

    if title.startswith('Finite'):
        sns.lineplot(rdf, x='n_tasks', y='dmmse_mse', color='purple', alpha=0.3, label='dMMSE', marker='o', markersize=8, linestyle='dashed', errorbar=None)
    else:
        sns.lineplot(rdf, x='n_tasks', y='dmmse_mse_unr', color='purple', alpha=0.3, label='dMMSE', marker='o', markersize=8, linestyle='dashed', errorbar=None)

    sns.lineplot(rdf, x='n_tasks', y='ridge_mse', color='r', alpha=0.5, label='Ridge', marker='o', markersize=8, linestyle='dashed', errorbar=None)

    g.legend()

    g.set_ylabel('MSE')
    g.set_xlabel('$k$')

    g.set_title(title)

    fig = g.get_figure()
    fig.tight_layout()
    return fig


fig = make_iwl_to_icl_plot('mse_pretrain', 'Finite Task Distribution', ylim=False, sim=True)
# fig.set_size_inches(8, 6)

fig.savefig('fig/final/varlen_reg_icl_pretrain_mse.svg')
# fig.savefig('fig/final/varlen_reg_icl_pretrain_mse.png')
fig.show()

# <codecell>
fig = make_iwl_to_icl_plot('mse_true', 'Unrestricted Task Distribution')
fig.savefig('fig/final/varlen_reg_icl_true_mse.svg')
# fig.set_size_inches(8, 6)
# fig.savefig('fig/final/varlen_true_mse.png')
fig.show()


# <codecell>
#### PATCH-WISE SCALING
df = collate_dfs('remote/12_icl_clean/varlen_scale_pd', show_progress=False)
mdf = df[df['name'] != 'Ridge']

# <codecell>
def extract_plot_vals(row):
    hist = row['hist']['test']
    slices = np.exp(np.linspace(-6, 0, num=10))
    slices = np.insert(slices, 0, 0)
    slice_idxs = (slices * len(hist)).astype(np.int32)
    slice_idxs[-1] -= 1  # adjust for last value
    
    row.test_task.autoregressive = False
    n_points = row.test_task.n_points
    row.test_task.var_length = False
    all_ridge_err = []
    for n in range(3, n_points+1):
        row.test_task.n_points = n
        xs, ys = next(row.test_task)
        ys_pred = estimate_ridge(None, *unpack(xs))
        ridge_err = np.mean((ys_pred - ys)**2)
        all_ridge_err.append(ridge_err)
    
    ridge_err = np.mean(all_ridge_err)
    hist_dict = {np.log10((idx + 1) * 1000): hist[idx]['loss'].item() - ridge_err for ratio, idx in zip(slices, slice_idxs)}
    
    return pd.Series([
        row['name'],
        row.train_task.n_dims,
        row.train_task.n_points,
        row['info']['mse'].item() - ridge_err,
        hist_dict,
    ], index=['name', 'n_dims', 'n_points', 'mse_final', 'hist'])

plot_df = mdf.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)

# plot_df[(plot_df['name'] == 'Transformer') & (plot_df['mse_final'] > 0.3)] = None  # remove failing seeds
# plot_df[(plot_df['name'] == 'MLP') & (plot_df['n_points'] == 64)] = None  # drop n = 64 MLP example
plot_df = plot_df.dropna()

stat_df, hist_df = plot_df.iloc[:,:-1], plot_df.iloc[:,-1]
hist_df = pd.DataFrame(hist_df.tolist())
hist_df

adf = pd.concat((stat_df, hist_df), axis=1) \
        .melt(id_vars=stat_df.columns, var_name='train_prop', value_name='mse')
adf

# <codecell>
# Full performance plots
def make_pd_plot(name):
    cdf = adf[adf['name'] == name]
    g = sns.FacetGrid(cdf, col='n_dims', height=2, aspect=1.25)
    g.map_dataframe(sns.lineplot, x='n_points', y='mse', hue='train_prop', marker='o', alpha=0.7)
    g.add_legend()

    g._legend.set_title('Train steps')

    for t in g._legend.texts:
        label = t.get_text()
        t.set_text('${10}^{%s}$' % label)

    for ax in g.axes.ravel():
        ax.set_xscale('log')
        n_dims = ax.get_title().split('=')[1]
        ax.set_title(f'$n = {n_dims}$')
        ax.set_ylabel('Excess MSE')
        ax.set_xlabel('$L$')
        ax.axhline(y=0.95, linestyle='dashed', color='k', alpha=0.3)

    g.tight_layout()
    return g

g = make_pd_plot('MLP')
# g.savefig('fig/final/fig_reg_icl_supp/reg_icl_mlp_pd.svg')
# <codecell>
g = make_pd_plot('Mixer')
# g.savefig('fig/final/fig_reg_icl_supp/reg_icl_mix_pd.svg')
# <codecell>
g = make_pd_plot('Transformer')
# g.savefig('fig/final/fig_reg_icl_supp/reg_icl_transf_pd.svg')

# <codecell>
# Subsection on high dimensions
adf = plot_df[plot_df['n_dims'] == 8]
g = sns.lineplot(adf, x='n_points', y='mse_final', hue='name', marker='o', estimator='mean', markersize=8, hue_order=['MLP', 'Mixer', 'Transformer'])
g.set_xscale('log', base=2)
g.axhline(0.95, linestyle='dashed', color='k', alpha=0.3)

g.set_ylabel('Excess MSE')
g.set_xlabel('$L$')
g.legend_.set_title(None)

fig = g.figure
# fig.set_size_inches(4, 3)
fig.tight_layout()
# fig.savefig('fig/final/varlen_reg_icl_scale_pd.svg')
# fig.savefig('fig/final/varlen_reg_icl_scale_pd.png')
