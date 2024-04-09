"""
Assembling the final relational task figures for NeurIPS 2024 cleanly
"""

# <codecell>
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, RfConfig, SpatialMlpConfig
from model.transformer import TransformerConfig
from task.function import PowerTask 

fig_dir = Path('fig/final')

# <codecell>
### SCALE PLOTS
df = collate_dfs('remote/14_relational_clean/scale')
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
        type(row.train_task).__name__,
        np.log10(row['info']['size']),
        row['info']['flops'],
        hist_dict,
    ], index=['name', 'task', 'log10_size', 'flops', 'hist'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)

plot_df

# <codecell>
def format_df(name=None, task=None):
    mdf = plot_df.copy()
    if task is not None:
        mdf = plot_df[(plot_df['task'] == task)].reset_index(drop=True)

    if name is not None:
        mdf = mdf[(mdf['name'] == name)].reset_index(drop=True)

    m_hist_df = pd.DataFrame(mdf['hist'].tolist())
    cols = [c for c in mdf.columns if c != 'hist']
    mdf = pd.concat((mdf[cols], m_hist_df), axis='columns') \
            .melt(id_vars=cols, var_name='hist_idx', value_name='loss')

    mdf['train_iters'] = (mdf['hist_idx'] + 1) * 1000   # 1k iterations per save 
    mdf['total_pflops'] = (mdf['flops'] * mdf['train_iters']) / 1e15
    return mdf


def plot_compute(df, title, hue_name='log10_size'):
    g = sns.lineplot(df, x='total_pflops', y='loss', hue=hue_name, marker='o', palette='flare_r', alpha=0.7, legend='brief')
    g.set_xscale('log')

    g.set_ylabel('Loss')
    g.set_xlabel('Compute (PFLOPs)')

    g.legend()
    g.legend_.set_title('# Params')

    for t in g.legend_.texts:
        label = t.get_text()
        t.set_text('${10}^{%s}$' % label)

    g.spines[['right', 'top']].set_visible(False)

    g.set_title(title)
    fig = g.get_figure()
    fig.tight_layout()
    return fig

mdf = format_df('MLP', 'RingMatch')
fig = plot_compute(mdf, 'MLP')
fig.savefig(fig_dir / 'match_mlp_scale.svg')

# <codecell>
mdf = format_df('MLP', 'FreeOddballTask')
fig = plot_compute(mdf, 'MLP')
fig.savefig(fig_dir / 'free_oddball_mlp_scale.svg')

# <codecell>
mdf = format_df('MLP', 'LineOddballTask')
fig = plot_compute(mdf, 'MLP')
fig.savefig(fig_dir / 'line_oddball_mlp_scale.svg')

# %%
mdf = format_df('Transformer', 'RingMatch')
fig = plot_compute(mdf, 'Transformer')
fig.savefig(fig_dir / 'match_transf_scale.svg')

# %%
mdf = format_df('Transformer', 'FreeOddballTask')
fig = plot_compute(mdf, 'Transformer')
fig.savefig(fig_dir / 'free_oddball_transf_scale.svg')

# %%
mdf = format_df('Transformer', 'LineOddballTask')
fig = plot_compute(mdf, 'Transformer')
fig.savefig(fig_dir / 'line_oddball_transf_scale.svg')

# %%
def plot_rbmlp(mdf):
    g = sns.lineplot(mdf, x='total_pflops', y='loss', marker='o', hue='log10_size', palette='flare_r', alpha=0.7)
    g.set_xscale('log')

    g.set_ylabel('Loss')
    g.set_xlabel('Compute (PFLOPs)')

    g.legend(labels=['$10^{1.6}$'])
    g.legend_.set_title('# Params')

    g.spines[['right', 'top']].set_visible(False)

    g.set_title('RB MLP')
    fig = g.get_figure()
    fig.tight_layout()
    return fig

mdf = format_df('RB MLP', 'RingMatch')
plot_rbmlp(mdf)
fig.savefig(fig_dir / 'match_rbmlp_scale.svg')

# <codecell>
mdf = format_df('RB MLP', 'FreeOddballTask')
plot_rbmlp(mdf)
fig.savefig(fig_dir / 'free_oddball_rbmlp_scale.svg')

# <codecell>
mdf = format_df('RB MLP', 'LineOddballTask')
plot_rbmlp(mdf)
fig.savefig(fig_dir / 'line_oddball_rbmlp_scale.svg')

# <codecell>
def plot_all(mdf):
    g = sns.scatterplot(mdf, x='total_pflops', y='loss', hue='name', marker='o', alpha=0.6, legend='auto')
    g.set_xscale('log')

    g.legend_.set_title(None)

    g.set_ylabel('Loss')
    g.set_xlabel('Compute (PFLOPs)')

    g.spines[['right', 'top']].set_visible(False)

    fig = g.get_figure()
    fig.tight_layout()
    return fig

mdf = format_df(task='RingMatch')
fig = plot_all(mdf)
fig.savefig(fig_dir / 'match_all_scale.svg')

# <codecell>
mdf = format_df(task='FreeOddballTask')
fig = plot_all(mdf)
fig.savefig(fig_dir / 'free_oddball_all_scale.svg')

# <codecell>
mdf = format_df(task='LineOddballTask')
fig = plot_all(mdf)
fig.savefig(fig_dir / 'line_oddball_all_scale.svg')

# <codecell>
### MATCH GENERALIZE
df = collate_dfs('remote/14_relational_clean/generalize')
df.iloc[0]['info']

# <codecell>
### SCALE ODDBALL
full_dicts = []

for _, row in df.iterrows():
    for key in row['info']:
        if key.startswith('acc'):
            _, is_scram, r = key.split('_')
            full_dicts.append({
                'name': row['name'],
                'train_scramble': row['info']['common_args']['scramble'],
                'test_scramble': is_scram,
                'radius': r,
                'acc': row['info'][key].item()
            })


plot_df = pd.DataFrame(full_dicts)
plot_df

# <codecell>
g = sns.catplot(plot_df, 
            x='radius', y='acc', hue='name', row='train_scramble', col='test_scramble', 
            kind='bar', aspect=2, height=2)

g.set_ylabels('Accuracy')
g.set_xlabels('Radius')
g._legend.set_title(None)

g.tight_layout()
g.savefig(fig_dir / 'match_generalize.svg')
