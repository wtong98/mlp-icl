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
df = collate_dfs('remote/14_relational_clean/scale_match')

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
    ], index=['name', 'log10_size', 'flops', 'hist'])

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

mdf = format_df('MLP')
fig = plot_compute(mdf, 'MLP')
fig.savefig(fig_dir / 'match_mlp_scale.svg')

# %%
mdf = format_df('Transformer')
fig = plot_compute(mdf, 'Transformer')
fig.savefig(fig_dir / 'match_transf_scale.svg')

# %%
mdf = format_df('RB MLP')
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
fig.savefig(fig_dir / 'match_rbmlp_scale.svg')

# <codecell>
# sigma = 3

# mdf = format_df('MLP')
# min_df = mdf.groupby('total_pflops', as_index=False).min()
# x_min = min_df['total_pflops']
# y_min = min_df['loss']
# y_min = gaussian_filter1d(y_min, sigma=sigma)

# max_df = mdf.groupby('total_pflops', as_index=False).max()
# x_max = max_df['total_pflops']
# y_max = max_df['loss']
# y_max = gaussian_filter1d(y_max, sigma=sigma)


# plt.plot(x_min, y_min, 'o--', alpha=0.5,)
# plt.plot(x_max, y_max, 'o--', alpha=0.5,)

# plt.scatter(mdf['total_pflops'], mdf['loss'], alpha=0.1)
# plt.xscale('log')


# <codecell>
mdf = format_df()
# g = sns.lineplot(mdf, x='total_pflops', y='loss', hue='name', alpha=0.6, ci=None)
g = sns.scatterplot(mdf, x='total_pflops', y='loss', hue='name', marker='o', alpha=0.6, legend='auto')
g.set_xscale('log')

g.legend_.set_title(None)

g.set_ylabel('Loss')
g.set_xlabel('Compute (PFLOPs)')

g.spines[['right', 'top']].set_visible(False)

fig = g.get_figure()
fig.tight_layout()
fig.savefig(fig_dir / 'match_all_scale.svg')