"""
Assembling the final simple task figures for NeurIPS 2024 cleanly
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
from task.function import PowerTask 

def plot_compute(df, title, hue_name='log10_size', legend='brief', raise10=True):
    g = sns.lineplot(df, x='total_pflops', y='mse', hue=hue_name, marker='o', palette='flare_r', alpha=0.7, legend=legend)
    g.axhline(0.05, linestyle='dashed', color='k', alpha=0.3)
    g.set_xscale('log')
    g.set_yscale('log')

    g.set_ylabel('MSE')
    g.set_xlabel('Compute (PFLOPs)')
    g.legend_.set_title('# Params')

    if raise10:
        for t in g.legend_.texts:
            label = t.get_text()
            t.set_text('${10}^{%s}$' % label)

    g.spines[['right', 'top']].set_visible(False)

    g.set_title(title)
    fig = g.get_figure()
    fig.tight_layout()
    return fig


def format_df(name=None, power=1):
    mdf = plot_df[plot_df['power'] == power]
    if name is not None:
        mdf = plot_df[(plot_df['name'] == name) & (plot_df['power'] == power)]
    mdf = mdf.reset_index(drop=True)

    m_hist_df = pd.DataFrame(mdf['hist'].tolist())
    mdf = pd.concat((mdf, m_hist_df), axis='columns') \
            .melt(id_vars=plot_df.columns, var_name='hist_idx', value_name='mse')

    mdf['train_iters'] = (mdf['hist_idx'] + 1) * 1000   # 1k iterations per save 
    mdf['total_pflops'] = (mdf['flops'] * mdf['train_iters']) / 1e15
    return mdf

fig_dir = Path('fig/final')

# <codecell>
### REGRESSION: scaling
df = collate_dfs('remote/13_simple_clean/scale')

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
        row.train_task.power,
        row.train_task.tokenize,
        np.log10(row['info']['size']),
        row['info']['flops'],
        hist_dict,
        f"{row['config']['n_layers']}-{row['config']['n_hidden']}",
    ], index=['name', 'n_dims', 'power', 'token_size', 'log10_size', 'flops', 'hist', 'arch'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
#             .melt(id_vars=['name', 'n_pretrain_tasks', 'n_dims'], var_name='mse_type', value_name='mse')

# <codecell>
for p in [1, 2, 3]:
    mdf = format_df('MLP', power=p)
    fig = plot_compute(mdf, 'MLP')
    fig.savefig(fig_dir / f'reg_p{p}_mlp_scale.svg')
    fig.show()
    fig.clf()

# <codecell>
for p in [1, 2, 3]:
    fig.clf()
    mdf = format_df('Transformer', power=p)
    fig = plot_compute(mdf, 'Transformer')
    fig.savefig(fig_dir / f'reg_p{p}_transf_scale.svg')
    fig.show()


# <codecell>
for p in [1, 2, 3]:
    mdf = format_df(power=p)
    g = sns.scatterplot(mdf, x='total_pflops', y='mse', hue='name', marker='o', alpha=0.7)
    g.axhline(0.05, linestyle='dashed', color='k', alpha=0.3)
    g.set_xscale('log')
    g.set_yscale('log')

    g.set_ylabel('MSE')
    g.set_xlabel('Compute (PFLOPs)')
    g.legend_.set_title('# Params')

    g.spines[['right', 'top']].set_visible(False)

    fig = g.get_figure()
    fig.tight_layout()
    fig.savefig(fig_dir / f'reg_p{p}_all_scale.svg')
    fig.show()
    fig.clf()

# <codecell>
### PER-TOKEN IMPROVEMENTS
df = collate_dfs('remote/13_simple_clean/tokenize')

# <codecell>

# <codecell>
for p in [1, 2, 3]:
    plot_df = df.apply(extract_plot_vals, axis=1) \
                .reset_index(drop=True)

    mdf = format_df('Transformer', power=p)
    mdf['token_size'] = np.log2(mdf['token_size'])
    
    g = sns.lineplot(mdf, x='total_pflops', y='mse', hue='token_size', marker='o', palette='flare_r', alpha=0.7, legend='full')
    g.axhline(0.05, linestyle='dashed', color='k', alpha=0.3)
    g.set_xscale('log')
    g.set_yscale('log')

    g.set_ylabel('MSE')
    g.set_xlabel('Compute (PFLOPs)')
    g.legend_.set_title('Token size')

    for t in g.legend_.texts:
        label = float(t.get_text())
        t.set_text(f'{int(2**label)}')

    g.spines[['right', 'top']].set_visible(False)

    # g.set_title('Transformer')
    fig = g.get_figure()
    fig.tight_layout()
    fig.savefig(fig_dir / f'reg_p{p}_transf_tokenize.svg')
    fig.clf()

