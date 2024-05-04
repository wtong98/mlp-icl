"""
Assembling the final relational task figures for NeurIPS 2024 cleanly
"""

# <codecell>
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, DotMlpConfig
from model.transformer import TransformerConfig
from task.match import RingMatch 
from task.oddball import FreeOddballTask, LineOddballTask

fig_dir = Path('fig/final')
set_theme()

# <codecell>
### SCALE PLOTS
df1 = collate_dfs('remote/14_relational_clean/scale_match')
df2 = collate_dfs('remote/14_relational_clean/scale_oddball')
df = pd.concat((df1, df2))

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
    fig.set_size_inches(4, 3)
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

# %%
mdf = format_df('RB MLP (deep)', 'LineOddballTask')
fig = plot_compute(mdf, 'RB MLP (deep)')
fig.savefig(fig_dir / 'line_oddball_rbmlp_deep_scale.svg')

# <codecell>
def plot_all(mdf, title=''):
    g = sns.scatterplot(mdf, x='total_pflops', y='loss', hue='name', marker='o', alpha=0.6, legend='auto', palette=['C0', 'C6', 'C2'], hue_order=['MLP', 'RB MLP', 'Transformer'])
    g.set_xscale('log')

    g.legend_.set_title(None)

    g.set_ylabel('Loss')
    g.set_xlabel('Compute (PFLOPs)')
    g.set_title(title)

    fig = g.get_figure()
    fig.tight_layout()
    return fig

mdf = format_df(task='RingMatch')
fig = plot_all(mdf, title='Match')
fig.savefig(fig_dir / 'fig3/match_all_scale.svg')

# <codecell>
mdf = format_df(task='FreeOddballTask')
fig = plot_all(mdf, 'Sphere Oddball')
fig.savefig(fig_dir / 'fig3/free_oddball_all_scale.svg')

# <codecell>
mdf = format_df(task='LineOddballTask')
g = sns.scatterplot(mdf, x='total_pflops', y='loss', hue='name', marker='o', alpha=0.6, legend='auto', palette=['C0', 'C6', 'C8', 'C2'], hue_order=['MLP', 'RB MLP', 'RB MLP (deep)', 'Transformer'])
g.set_xscale('log')

g.legend_.set_title(None)

g.set_ylabel('Loss')
g.set_xlabel('Compute (PFLOPs)')
g.set_title('Line Oddball')

fig = g.get_figure()
fig.tight_layout()

fig.savefig(fig_dir / 'fig3/line_oddball_all_scale.svg')

# <codecell>
### MATCH GENERALIZE
df = collate_dfs('remote/14_relational_clean/generalize')
df.iloc[0]['info']

# <codecell>
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
radii = np.unique(plot_df['radius'])
r = radii[0]
task = RingMatch(radius=r)
# TODO: make model with perfect success


# <codecell>
g = sns.catplot(plot_df, 
            x='radius', y='acc', hue='name', row='train_scramble', col='test_scramble', 
            kind='bar', aspect=2.5, height=1.25, palette=['C0', 'C6', 'C2'], hue_order=['MLP', 'RB MLP', 'Transformer'])

g.set_ylabels('Accuracy')
g.set_xlabels('Radius')
g._legend.set_title(None)

g.tight_layout()
g.savefig(fig_dir / 'fig3/match_generalize.svg')

# <codecell>
### ODDBALL GENERALIZE
df = collate_dfs('remote/14_relational_clean/generalize_oddball')
df

# <codecell>
fo_dicts = []

for _, row in df.iterrows():
    if type(row.train_task).__name__ != 'FreeOddballTask':
        continue

    for key in row['info']:
        if key.startswith('acc'):
            _, _, d = key.split('_')
            fo_dicts.append({
                'name': row['name'],
                'distance': d,
                'acc': row['info'][key].item()
            })


plot_df = pd.DataFrame(fo_dicts)
plot_df

# <codecell>
### Free oddball perfect solution
dists = np.unique(plot_df['distance'])
res = {}

for d in dists:
    d = float(d)
    task = FreeOddballTask(discrim_dist=d, batch_size=8192)
    xs, ys = next(task)

    xs = xs - np.mean(xs, axis=1, keepdims=True)
    out = np.linalg.norm(xs, axis=-1)
    ys_pred = np.argmax(out, axis=-1)
    res[d] = np.mean(ys_pred == ys)

# <codecell>
g = sns.barplot(plot_df, x='distance', y='acc', hue='name', hue_order=['MLP', 'RB MLP', 'Transformer'], palette=['C0', 'C6', 'C2'])

bar_width = 0.4

for tick, lab in zip(g.get_xticks(), g.get_xticklabels()):
    r = lab.get_text()
    optim_res = res[float(r)]
    
    if tick == 0:
        g.hlines(y=optim_res, xmin=tick-bar_width, xmax=tick+bar_width, linestyle='dashed', color='r', alpha=0.7, label='Optimal')
    else:
        g.hlines(y=optim_res, xmin=tick-bar_width, xmax=tick+bar_width, linestyle='dashed', color='r', alpha=0.7)

g.legend()
g.legend_.set_title(None)

g.set_xlabel('Distance')
g.set_ylabel('Accuracy')

g.spines[['top', 'right']].set_visible(False)
fig = g.figure
fig.set_size_inches(4, 1.25)
fig.savefig(fig_dir / 'fig3/free_oddball_generalize.svg')

# <codecell>
### PLOT LOGITS PER DISTANCE
model = MlpConfig(**df.iloc[0]['config']).to_model()
params = df.iloc[0]['state']['params']

# <codecell>
res_dicts = []
n_iters = 5
n_points = 20

config_types = [MlpConfig, TransformerConfig, DotMlpConfig]
for config_type, (_, row) in zip(config_types, df.iloc[:3].iterrows()):
    model = config_type(**row['config']).to_model()
    params = row['state']['params']

    for _ in range(n_iters):
        test_cluster = np.random.randn(n_points, 5, 2)
        target_points = np.linspace(5, 25, num=n_points)
        target = np.zeros((n_points, 1, 2))
        target[np.arange(n_points),0,1] = target_points
        points = np.concatenate((test_cluster, target), axis=1)

        logits = model.apply({'params': params}, points)
        res_dicts.extend([
            {
                'name': row['name'],
                'distance': d,
                'logit': l.item()
            } 
        for d, l in zip(target_points, logits[:,-1])])

plot_df = pd.DataFrame(res_dicts)
plot_df

# <codecell>
g = sns.lineplot(plot_df, x='distance', y='logit', hue='name', marker='o', alpha=0.7, hue_order=['MLP', 'RB MLP', 'Transformer'], palette=['C0', 'C6', 'C2'])

xs = np.linspace(5, 25, num=n_points)
g.plot(xs, xs**2 * 0.55, '--', color='k', alpha=0.7)
g.plot(xs, xs * 2.1, '--', color='k', alpha=0.7)
g.plot(xs, 0 * xs + 9.1, '--', color='k', alpha=0.7)

g.set_xscale('log')
g.set_yscale('log')

g.set_xlabel('Distance')
g.set_ylabel('Logit')

g.legend()
g.legend_.set_title(None)

g.spines[['top', 'right']].set_visible(False)
fig = g.figure
fig.set_size_inches(4, 3)
fig.tight_layout()
fig.savefig(fig_dir / 'fig3/free_oddball_logit.svg')

# <codecell>
lo_dicts = []

for _, row in df.iterrows():
    if type(row.train_task).__name__ != 'LineOddballTask':
        continue

    train_dist = row['train_task'].perp_dist

    for key in row['info']:
        if key.startswith('acc'):
            _, _, d = key.split('_')
            lo_dicts.append({
                'name': row['name'],
                'train_distance': train_dist,
                'test_distance': d,
                'acc': row['info'][key].item()
            })


plot_df = pd.DataFrame(lo_dicts)
plot_df

# <codecell>
### Line oddball max solution
dists = np.unique(plot_df['test_distance'])
dist_res = {}

for d in dists:
    d = float(d)
    task = LineOddballTask(perp_dist=d, batch_size=8192)
    xs, ys = next(task)

    xs = xs - np.mean(xs, axis=1, keepdims=True)
    out = np.linalg.norm(xs, axis=-1)
    ys_pred = np.argmax(out, axis=-1)
    dist_res[d] = np.mean(ys_pred == ys)


# <codecell>
### Line oddball reg solution
dists = np.unique(plot_df['test_distance'])
reg_res = {}

for d in dists:
    d = float(d)
    task = LineOddballTask(perp_dist=d, batch_size=8192)
    xs, ys = next(task)

    # xs = xs - np.mean(xs, axis=1, keepdims=True)

    x = xs[:,:,[0]]
    y = xs[:,:,[1]]

    res = x @ np.linalg.pinv(t(x) @ x) @ t(x) @ y
    out = ((res - y)**2).squeeze()
    reg_res[d] = np.mean(out.argmax(axis=-1) == ys)

reg_res
# <codecell>
g = sns.catplot(plot_df, x='test_distance', y='acc', col='train_distance', hue='name', kind='bar', height=1.5, aspect=2.4, legend_out=True, palette=['C0', 'C6', 'C8', 'C2'], hue_order=['MLP', 'RB MLP', 'RB MLP (deep)', 'Transformer'])

handle = None
for ax in g.axes.ravel():
    bar_width = 0.4

    for tick, lab in zip(ax.get_xticks(), ax.get_xticklabels()):
        r = lab.get_text()
        optim_res = dist_res[float(r)]
        r_res = reg_res[float(r)]
        
        ax.hlines(y=optim_res, xmin=tick-bar_width, xmax=tick+bar_width, linestyle='dashed', color='r', alpha=0.7)
        # ax.hlines(y=r_res, xmin=tick-bar_width, xmax=tick+bar_width, linestyle='dashed', color='m', alpha=0.7)

g._legend.set_title(None)
g.set_xlabels('Test $d$')
g.set_ylabels('Accuracy')
g.set_titles('Train $d$ = {col_name}')

g.tight_layout()
g.savefig(fig_dir / 'fig3/line_oddball_generalize.svg')

# %%
