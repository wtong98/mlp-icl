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
fig.savefig(fig_dir / 'reg_icl_mlp_scale.svg')
fig.show()

# <codecell>
mdf = format_df('Mixer')
fig = plot_compute(mdf, 'Mixer')
fig.savefig(fig_dir / 'reg_icl_mix_scale.svg')
fig.show()

# <codecell>
mdf = format_df('Transformer')
fig = plot_compute(mdf, 'Transformer')
fig.savefig(fig_dir / 'reg_icl_transf_scale.svg')
fig.show()

# <codecell>
mdf = format_df()

g = sns.scatterplot(mdf, x='total_pflops', y='mse', hue='name', marker='o', alpha=0.6, legend='auto')
g.set_xscale('log')
g.axhline(ridge_result, linestyle='dashed', color='k', alpha=0.3)

g.legend_.set_title(None)

g.set_ylabel('MSE')
g.set_xlabel('Compute (PFLOPs)')
g.set_title('ICL Regression')

g.spines[['right', 'top']].set_visible(False)

fig = g.get_figure()
fig.tight_layout()
fig.savefig(fig_dir / 'fig1/reg_icl_all_scale.svg')

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
def make_iwl_to_icl_plot(mse_type, title=''):
    g = sns.lineplot(mdf, x='n_tasks', y=mse_type, hue='name', marker='o')
    g.set_xscale('log')
    g.plot(ddf['n_tasks'], ddf[mse_type], linestyle='dashed', color='purple', alpha=0.3, label='dMMSE', marker='o', markersize=5)
    g.axhline(y=ridge_result, linestyle='dashed', color='k', alpha=0.3, label='Ridge')

    g.legend()

    g.set_ylabel('MSE')
    g.set_xlabel('# Pretrain Tasks')

    g.spines[['right', 'top']].set_visible(False)
    g.set_title(title)

    fig = g.get_figure()
    fig.tight_layout()
    return fig

fig = make_iwl_to_icl_plot('mse_pretrain', 'Pretraining Distribution')
fig.savefig('fig/final/fig1/reg_icl_pretrain_mse.svg')
fig.clf()

fig = make_iwl_to_icl_plot('mse_true', 'True Distribution')
fig.savefig('fig/final/fig1/reg_icl_true_mse.svg')


# <codecell>
#### PATCH-WISE SCALING
df = collate_dfs('remote/12_icl_clean/scale_pd/')
mdf = df[df['name'] != 'Ridge']

# <codecell>
def extract_plot_vals(row):
    hist = row['hist']['test']
    slices = np.exp(np.linspace(-6, 0, num=10))
    slices = np.insert(slices, 0, 0)
    slice_idxs = (slices * len(hist)).astype(np.int32)
    slice_idxs[-1] -= 1  # adjust for last value
    
    xs, ys = next(row.test_task)
    ys_pred = estimate_ridge(None, *unpack(xs))
    ridge_err = np.mean((ys_pred - ys)**2)

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
    g = sns.FacetGrid(cdf, col='n_dims')
    g.map_dataframe(sns.lineplot, x='n_points', y='mse', hue='train_prop', marker='o', alpha=0.7)
    g.add_legend()

    g._legend.set_title('Train steps')

    for t in g._legend.texts:
        label = t.get_text()
        t.set_text('${10}^{%s}$' % label)

    for ax in g.axes.ravel():
        ax.set_xscale('log')
        n_dims = ax.get_title().split('=')[1]
        ax.set_title(f'D = {n_dims}')
        ax.set_ylabel('Excess MSE')
        ax.set_xlabel('# Points')
        ax.axhline(y=0.95, linestyle='dashed', color='k', alpha=0.3)

    g.tight_layout()
    return g

g = make_pd_plot('MLP')
g.savefig('fig/final/reg_icl_mlp_pd.svg')

g = make_pd_plot('Mixer')
g.savefig('fig/final/reg_icl_mix_pd.svg')

g = make_pd_plot('Transformer (softmax)')  # TODO: rename
g.savefig('fig/final/reg_icl_transf_pd.svg')

# <codecell>
# Subsection on high dimensions
adf = plot_df[plot_df['n_dims'] == 8]
g = sns.lineplot(adf, x='n_points', y='mse_final', hue='name', marker='o')
g.set_xscale('log', base=2)
g.axhline(0.95, linestyle='dashed', color='k', alpha=0.3)

g.set_ylabel('MSE')
g.set_xlabel('# Points in context')
g.legend_.set_title(None)
g.spines[['right', 'top']].set_visible(False)

fig = g.figure
fig.tight_layout()
fig.savefig('fig/final/fig1/reg_icl_excess_mse.svg')

# <codecell>
############################
### CLASSIFICATION PLOTS ###
############################


### Classification scale plots
df = collate_dfs('remote/12_icl_clean/cls_scale')
df

# <codecell>
def extract_plot_vals(row):
    hist = row['hist']['test']
    slices = np.exp(np.linspace(-6, 0, num=10))
    slices = np.insert(slices, 0, 0)
    slice_idxs = (slices * len(hist)).astype(np.int32)
    slice_idxs[-1] -= 1  # adjust for last value
    
    hist_dict = {idx: hist[idx]['loss'].item() for idx in slice_idxs}

    return pd.Series([
        row['name'],
        np.log10(row['info']['size']),
        row['info']['flops'],
        hist_dict,
        f"{row['config']['n_layers']}-{row['config']['n_hidden']}",
    ], index=['name', 'log10_size', 'flops', 'hist', 'arch'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
def format_df(name=None):
    mdf = plot_df.copy()
    if name is not None:
        mdf = plot_df[plot_df['name'] == name].reset_index(drop=True)

    m_hist_df = pd.DataFrame(mdf['hist'].tolist())
    mdf = pd.concat((mdf[['name', 'flops', 'log10_size', 'arch']], m_hist_df), axis='columns') \
            .melt(id_vars=['name', 'flops', 'log10_size', 'arch'], var_name='hist_idx', value_name='loss')

    mdf['train_iters'] = (mdf['hist_idx'] + 1) * 1000   # 1k iterations per save 
    mdf['total_pflops'] = (mdf['flops'] * mdf['train_iters']) / 1e15
    return mdf


def plot_compute(df, title, hue_name='log10_size'):
    g = sns.lineplot(df, x='total_pflops', y='loss', hue=hue_name, marker='o', palette='flare_r', alpha=0.5, legend='brief')
    g.set_xscale('log')

    g.set_ylabel('Loss')
    g.set_xlabel('Compute (PFLOPs)')
    g.legend_.set_title('# Params')

    for t in g.legend_.texts:
        label = t.get_text()
        t.set_text('${10}^{%s}$' % label)

    g.spines[['right', 'top']].set_visible(False)

    g.set_title(title)
    fig = g.get_figure()
    fig.set_size_inches(4, 3)
    fig.tight_layout()
    return fig

# <codecell>
mdf = format_df('MLP')
fig = plot_compute(mdf, 'MLP')
# fig.savefig(fig_dir / 'cls_icl_mlp_scale.svg')
fig.show()

# <codecell>
mdf = format_df('Mixer')
fig = plot_compute(mdf, 'Mixer')
# fig.savefig(fig_dir / 'cls_icl_mix_scale.svg')
fig.show()

# <codecell>
mdf = format_df('Transformer')
fig = plot_compute(mdf, 'Transformer')
# fig.savefig(fig_dir / 'cls_icl_transf_scale.svg')
fig.show()

# <codecell>
mdfs = [format_df('MLP'), format_df('Mixer'), format_df('Transformer')]
mdf = pd.concat(mdfs)
mdf = mdf[::2]

g = sns.scatterplot(mdf, x='total_pflops', y='loss', hue='name', marker='o', alpha=0.4, hue_order=('MLP','Mixer', 'Transformer'))
g.set_xscale('log')

g.legend_.set_title(None)

g.set_ylabel('Loss')
g.set_xlabel('Compute (PFLOPs)')

g.spines[['right', 'top']].set_visible(False)
g.set_title('ICL Classification')

fig = g.get_figure()
fig.set_size_inches(4, 3)
fig.tight_layout()
# fig.savefig(fig_dir / 'fig1/cls_icl_all_scale.svg')

# <codecell>
### CLS PLOT IWL --> ICL transition
df = collate_dfs('remote/12_icl_clean/cls_iwl_to_icl/')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].bursty,
        row['train_task'].n_classes,
        row['info']['iwl_acc'].item(),
        row['info']['icl_resamp_acc'].item(),
        row['info']['icl_swap_acc'].item(),
    ], index=['name', 'bursty', 'n_classes', 'iwl_acc', 'icl_resamp_acc', 'icl_swap_acc'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .melt(id_vars=['name', 'bursty', 'n_classes'], var_name='acc_type', value_name='acc') \
            .reset_index(drop=True)
plot_df

# <codecell>
g = sns.FacetGrid(plot_df, row='bursty', col='name', col_order=['MLP', 'Mixer', 'Transformer'], height=1.75, aspect=1.25)
g.map_dataframe(sns.lineplot, x='n_classes', y='acc', hue='acc_type', marker='o', palette='Paired', hue_order=['icl_swap_acc', 'icl_resamp_acc', 'iwl_acc'])

for ax in g.axes.ravel():
    ax.set_xscale('log', base=2)

g.set_ylabels('Accuracy')
g.set_xlabels('# Classes')
g.set_titles('{col_name}: $B = {row_name}$')

g.add_legend(title='Test Task')
label_to_name = {
    'icl_swap_acc': 'ICL (swap)',
    'icl_resamp_acc': 'ICL (resample)',
    'iwl_acc': 'IWL'
}
for t in g._legend.texts:
    label = t.get_text()
    t.set_text(label_to_name[label])

g.tight_layout()
g.savefig(fig_dir / 'cls_icl_transition_loss.svg')

# <codecell>
mdf = plot_df[plot_df['bursty'] == 4]

g = sns.FacetGrid(mdf, col='name', col_order=['MLP', 'Mixer', 'Transformer'], height=1.8, aspect=1.25)
g.map_dataframe(sns.lineplot, x='n_classes', y='acc', hue='acc_type', marker='o', palette='Paired', hue_order=['icl_swap_acc', 'icl_resamp_acc', 'iwl_acc'])

for ax in g.axes.ravel():
    ax.set_xscale('log', base=2)

g.set_ylabels('Accuracy')
g.set_xlabels('# Classes')
g.set_titles('{col_name}')

g.add_legend(title='Test Task')
label_to_name = {
    'icl_swap_acc': 'ICL (swap)',
    'icl_resamp_acc': 'ICL (resample)',
    'iwl_acc': 'IWL'
}
for t in g._legend.texts:
    label = t.get_text()
    t.set_text(label_to_name[label])

g.tight_layout()
g.savefig(fig_dir / 'fig1/cls_icl_b_4_transition_loss.svg')


# <codecell>
#### PATCH-WISE SCALING
df = collate_dfs('remote/12_icl_clean/cls_scale_pd/')
df

# <codecell>
def extract_plot_vals(row):
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
        row['info']['loss'].item(),
        hist_dict,
    ], index=['name', 'n_dims', 'n_points', 'final_loss', 'hist'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)

plot_df

# <codecell>
def format_df(name=None):
    mdf = plot_df.copy()
    if name is not None:
        mdf = plot_df[plot_df['name'] == name].reset_index(drop=True)

    m_hist_df = pd.DataFrame(mdf['hist'].tolist())
    cols = [c for c in mdf.columns if c != 'hist']
    mdf = pd.concat((mdf[cols], m_hist_df), axis='columns') \
            .melt(id_vars=cols, var_name='hist_idx', value_name='loss')

    mdf['train_iters'] = (mdf['hist_idx'] + 1) * 1000   # 1k iterations per save 
    mdf['log10_train_iters'] = np.log10(mdf['train_iters'].astype('float'))
    return mdf


def make_pd_plot(mdf):
    g = sns.FacetGrid(mdf, col='n_dims')
    g.map_dataframe(sns.lineplot, x='n_points', y='loss', hue='log10_train_iters', marker='o', alpha=0.7, legend='brief')
    g.add_legend()

    g._legend.set_title('Train steps')

    for t in g._legend.texts:
        label = t.get_text()
        t.set_text('${10}^{%s}$' % label)

    for ax in g.axes.ravel():
        ax.set_xscale('log', base=2)
        n_dims = ax.get_title().split('=')[1]
        ax.set_title(f'D = {n_dims}')
        ax.set_ylabel('Loss')
        ax.set_xlabel('# Points')

    g.tight_layout()
    return g

# <codecell>
mdf = format_df('MLP')
g = make_pd_plot(mdf)
# g.savefig(fig_dir / 'cls_icl_mlp_pd.svg')

# <codecell>
mdf = format_df('Mixer')
g = make_pd_plot(mdf)
# g.savefig(fig_dir / 'cls_icl_mix_pd.svg')

# <codecell>
mdf = format_df('Transformer')
g = make_pd_plot(mdf)
# g.savefig(fig_dir / 'cls_icl_transf_pd.svg')


# <codecell>
# TODO: record correct last value and redo
mdf = format_df()
mdf = mdf.dropna()
mdf.groupby('name').max()['hist_idx']

# <codecell>
mdf1 = mdf[(mdf['n_dims'] == 8) & (mdf['hist_idx'] == 15)]
mdf2 = mdf[(mdf['n_dims'] == 8) & (mdf['hist_idx'] == 63)]
mdf = pd.concat((mdf1, mdf2))
mdf

# <codecell>
mdfs = [format_df('MLP'), format_df('Mixer'), format_df('Transformer')]
mdf = pd.concat(mdfs)
mdf = mdf[(mdf['n_dims'] == 2)]
mdf

# <codecell>
g = sns.lineplot(mdf, x='n_points', y='final_loss', hue='name', marker='o', alpha=0.7, legend='brief', hue_order=['MLP', 'Mixer', 'Transformer'])
g.legend_.set_title(None)

g.set_xscale('log', base=2)
g.set_ylabel('Loss')
g.set_xlabel('# Points in context')

g.spines[['top', 'right']].set_visible(False)

fig = g.figure
fig.tight_layout()
# fig.savefig(fig_dir / 'fig1/cls_icl_all_pd.svg')
