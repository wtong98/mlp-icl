"""
Experimenting with the match tasks, which is comparable to the
delayed match-to-sample task of the Griffiths papeer.

Observations:

- MLP learns this in-context
- Increasing radius is no problem --> performs some sort of dot product operation
- All models struggle to generalize from fixed --> scramble task
    - MNN seems to perform marginally better
    --> uses ring structure, not purely dot-product-driven

Tentative conclusion: with perfect data, in-context learning is well
within capacity of MLP (among other models)
"""

# <codecell>
import functools

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.optimize import minimize

from common import *

import sys
sys.path.append('../')
from train import train
from model.knn import KnnConfig
from model.mlp import MlpConfig, DotMlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.match import RingMatch, LabelRingMatch

# <codecell>
# W/W_OUT SCRAMBLE GENERALIZATION
n_iters = 1
n_out = 6
train_iters = 20_000

common_args = {'n_points': n_out}
scramble_args = dict(scramble=True, **common_args)

all_cases = []
for _ in range(n_iters):
    for args in [common_args, scramble_args]:
        curr_cases = [
            Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=256)),
            Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=256, n_mlp_layers=3, pos_emb=True))
        ]

        for c in curr_cases:
            c.train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'}
            c.train_task = RingMatch(**args)
            c.test_task = RingMatch(batch_size=1024, **args)
            c.info['task_args'] = args
        
        all_cases.extend(curr_cases)


for case in tqdm(all_cases):
    print(case.name)
    case.run()

# <codecell>
eval_cases(all_cases, key_name='in_dist_eval', eval_task=RingMatch(batch_size=1024))
eval_cases(all_cases, key_name='radius_eval', eval_task=RingMatch(radius=2, batch_size=1024))
eval_cases(all_cases, key_name='scramble_eval', eval_task=RingMatch(scramble=True, batch_size=1024))

# <codecell>
df = pd.DataFrame(all_cases)

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['task_args'].get('scramble', False),
        row['info']['in_dist_eval'].astype(float),
        row['info']['radius_eval'].astype(float),
        row['info']['scramble_eval'].astype(float),
    ], index=['name', 'scramble', 'in_dist_acc', 'radius_acc', 'scramble_acc'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .melt(id_vars=['name', 'scramble'], var_name='acc_type', value_name='acc')

plot_df['acc'] = plot_df['acc'].astype('float')
plot_df

# <codecell>
g = sns.catplot(plot_df, x='scramble', y='acc', col='acc_type', hue='name', kind='bar')

# plt.tight_layout()
plt.savefig('fig/match_gen_simple.png')

# <codecell>
### RADIUS GENERALIZATION ON PLAIN MATCH TASK
n_iters = 1
n_out = 6
train_iters = 5_000

common_args = {'n_points': n_out, 'scramble': True}
# scramble_args = dict(scramble=True, **common_args)

all_cases = []
for _ in range(n_iters):
    for args in [common_args]:
        curr_cases = [
            # Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=16)),
            Case('Transformer', TransformerConfig(n_out=n_out, n_layers=8, n_hidden=256, n_mlp_layers=3, pos_emb=True))
        ]

        for c in curr_cases:
            c.train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'ce'}
            c.train_task = RingMatch(**args)
            c.test_task = RingMatch(batch_size=1024, **args)
            c.info['task_args'] = args
        
        all_cases.extend(curr_cases)


for case in tqdm(all_cases):
    print('TRAINING', case.name)
    case.run()

# <codecell>
eval_cases(all_cases, key_name='radius_eval', eval_task=RingMatch(radius=2, batch_size=1024))

df = pd.DataFrame(all_cases)
df['info'].tolist()

# <codecell>
### PLOTTING POTENTIAL SCALING IN RADIUS GENERALIZATION
pkl_path = Path('remote/4_match_tasks/scale')
dfs = [pd.read_pickle(f) for f in pkl_path.iterdir() if f.suffix == '.pkl']
df = pd.concat(dfs)

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_args']['train_iters'],
        row['config']['n_layers'],
        row['config']['n_hidden'],
        row['info']['size'],
        f"{row['config']['n_layers']}-{row['config']['n_hidden']}",
        row['info']['radius_0.5'].item(),
        row['info']['radius_1'].item(),
        row['info']['radius_2'].item(),
        row['hist']['train'][-1]['loss'].item() if len(row['hist']['train']) > 0 else None
    ], index=['name', 'train_iters', 'depth', 'width', 'size', 'arch', 0.5, 1, 2, 'loss'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            # .melt(id_vars=['name', 'train_iters', 'depth', 'width', 'size', 'arch'], value_vars=[0.5, 1, 2], var_name='radius', value_name='acc')
plot_df

# <codecell>
def scaling_law(x, consts):
    a, b, c, d = consts
    return a + b * (x + c)**(-d)

def get_law(target_df, feat, response='mse'):
    def loss(consts):
        result = scaling_law(target_df[feat], consts)
        return np.mean((result - target_df[response])**2)

    out = minimize(loss, np.zeros(4))
    return out


curr_df = plot_df[(plot_df['name'] == 'MLP') & (~pd.isna(plot_df['loss']))]
td = curr_df[curr_df['train_iters'] == 512000]
out = get_law(td, 'size', 'loss')
print(out)

g = sns.lineplot(curr_df, x='size', y='loss', hue='train_iters', markers=True, marker='o')

xs = np.sort(td['size'])
g.plot(xs, scaling_law(xs, out.x), '--', color='red')
a, b, c, d = out.x
g.text(10**5, 5e-2, fr'${a:.2f} + {b:.2f} (x + {c:.2f})^\wedge (-{d:.2f})$', color='red')

g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
plt.savefig('fig/match_scale_mlp_sizewise_loss.png')
plt.show()

# <codecell>
curr_df = plot_df[(plot_df['name'] == 'MLP') & (~pd.isna(plot_df['loss']))]
td = curr_df[curr_df['arch'] == '2-1024']
out = get_law(td, 'train_iters', 'loss')
print(out)

g = sns.lineplot(curr_df, x='train_iters', y='loss', hue='size', markers=True, marker='o')

xs = np.sort(td['train_iters'])
g.plot(xs, scaling_law(xs, out.x), '--', color='red')
a, b, c, d = out.x
g.text(10**5, 5e-2, fr'${a:.2f} + {b:.2f} (x + {c:.2f})^\wedge (-{d:.2f})$', color='red')

g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
plt.savefig('fig/match_scale_mlp_iterwisewise_loss.png')
plt.show()

# <codecell>
curr_df = plot_df[(plot_df['name'] == 'MLP') & (~pd.isna(plot_df['loss']))]
g = sns.lineplot(curr_df, x='train_iters', y='loss', hue='arch', markers=True, marker='o')

g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
plt.savefig('fig/match_scale_mlp_archwisewise_loss.png')
plt.show()

# <codecell>
curr_df = plot_df[(plot_df['name'] == 'Transformer') & (~pd.isna(plot_df['loss']))]
g = sns.lineplot(curr_df, x='size', y='loss', hue='train_iters', markers=True, marker='o')

# td = curr_df[curr_df['train_iters'] == 512000]
# out = get_law(td, 'size', 'loss')
# print(out)
# xs = np.sort(td['size'])
# g.plot(xs, scaling_law(xs, out.x), '--', color='red')
# a, b, c, d = out.x
# g.text(10**5, 5e-2, fr'${a:.2f} + {b:.2f} (x + {c:.2f})^\wedge (-{d:.2f})$', color='red')

g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
plt.savefig('fig/match_scale_transf_sizewise_loss.png')
plt.show()

# <codecell>
curr_df = plot_df[(plot_df['name'] == 'Transformer') & (~pd.isna(plot_df['loss']))]
g = sns.lineplot(curr_df, x='train_iters', y='loss', hue='size', markers=True, marker='o')

td = curr_df[curr_df['arch'] == '4-64']
out = get_law(td, 'train_iters', 'loss')
print(out)
xs = np.sort(td['train_iters'])
g.plot(xs, scaling_law(xs, out.x), '--', color='red')
a, b, c, d = out.x
g.text(10**5, 5e-2, fr'${a:.2f} + {b:.2f} (x + {c:.2f})^\wedge (-{d:.2f})$', color='red')

g.set_yscale('log')
g.set_xscale('log')

plt.tight_layout()
plt.savefig('fig/match_scale_transf_iterwise_loss.png')
plt.show()

# <codecell>
curr_df = plot_df[(plot_df['name'] == 'Transformer') & (~pd.isna(plot_df['loss']))]
g = sns.lineplot(curr_df, x='train_iters', y='loss', hue='arch', markers=True, marker='o')
g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
plt.savefig('fig/match_scale_transf_archwise_loss.png')
plt.show()

# <codecell>
curr_df = plot_df[plot_df['name'] == 'MLP']
g = sns.catplot(data=curr_df, x='size', y='loss', hue='train_iters', col='radius', kind='point', height=3, aspect=1.25)
plt.savefig('fig/match_scale_mlp_sizewise.png')
plt.show()

# <codecell>
form = 'size'

curr_df = plot_df[plot_df['name'] == 'MLP']
g = sns.catplot(data=curr_df, x='train_iters', y='acc', hue=form, col='radius', kind='point', height=3, aspect=1.25)
plt.savefig(f'fig/match_scale_mlp_{form}wise.png')
plt.show()

# <codecell>
curr_df = plot_df[plot_df['name'] == 'Transformer']
g = sns.catplot(data=curr_df, x='size', y='acc', hue='train_iters', col='radius', kind='point', height=3, aspect=1.25)
plt.savefig('fig/match_scale_tranf_sizewise.png')
plt.show()

# <codecell>
form = 'size'

curr_df = plot_df[plot_df['name'] == 'Transformer']
g = sns.catplot(data=curr_df, x='train_iters', y='acc', hue=form, col='radius', kind='point', height=3, aspect=1.25)
plt.savefig(f'fig/match_scale_transf_{form}wise.png')
plt.show()


# <codecell>
# WIDTH PERFORMANCE
n_iters = 3
n_out = 6
widths = [32, 64, 128, 256, 512, 1024]

common_args = {'train_iters': 20_000}

all_cases = []
for _ in range(n_iters):
    for w in widths:
        all_cases.extend([
            Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=w), match_experiment, experiment_args=common_args),
            Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=w, use_mlp_layers=True, pos_emb=True), match_experiment, experiment_args=common_args),
            Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=w), match_experiment, experiment_args=common_args),
        ])

for case in tqdm(all_cases):
    case.run()

# <codecell>
eval_cases(all_cases, key_name='scramble_eval', eval_task=RingMatch(scramble=True, batch_size=1024))

# <codecell>
df = pd.DataFrame(all_cases)

def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['config']['n_hidden'],
        row['info']['scramble_eval'].item(),
    ], index=['name', 'width', 'acc'])

df = df.apply(extract_plot_vals, axis=1)

# <codecell>
plt.gcf().set_size_inches(8, 3)
sns.barplot(df, x='width', y='acc', hue='name')
plt.tight_layout()
plt.savefig('fig/match_width_scramble.png')


# <codecell>
### KNN Experimentation
n_out = 6

def compute_dists(point, data):
    diff = data - point
    dists = np.sqrt(np.diag(diff @ diff.T))
    return dists

task = RingMatch(data_size=128, n_points=n_out)
model = MlpConfig(n_layers=1, n_hidden=32, n_out=n_out)

state, hist = train(model, task, loss='ce', train_iters=20_000, test_every=1_000)
# <codecell>
full_task = RingMatch(n_points=n_out, batch_size=1024)
xs, ys = next(full_task)
logits = state.apply_fn({'params': state.params}, xs)
mlp_probs = jax.nn.softmax(logits, axis=-1)
mlp_acc = np.mean(logits.argmax(-1) == ys)

data = task.data[0].reshape(task.data_size, -1)
labs = task.data[1]
knn = KnnConfig(beta=20, n_classes=n_out, xs=data, ys=labs).to_model()

xs_knn = xs.reshape(1024, -1)
knn_probs = knn(xs_knn)
knn_preds = knn_probs.argmax(-1)

knn_acc = np.mean(knn_preds == ys)

print('MLP', mlp_acc)
print('KNN', knn_acc)


plt.plot(knn_probs[0], 'o--')
plt.plot(mlp_probs[0], 'o--')



# <codecell>
logits = state.apply_fn({'params': state.params}, xs)

x = xs[0].flatten()
y = ys[0]

data = task.data[0].reshape(task.data_size, -1)

dists = compute_dists(x, data)
labs = task.data[1]

weights = np.exp(-3 * dists)
res = np.zeros(n_out)

for idx in range(n_out):
    res[idx] = np.sum(weights[labs == idx])
    
res = np.log(res + 1e-1)
plt.plot(res, 'o--')
ax = plt.twinx()
ax.plot(logits[0], 'o--', color='red')

print(labs[np.argmin(dists)])



# <codecell>
### LABEL RING MATCH TASK

# TODO: test new seed paradigm
n_iters = 3
n_out = 6

def make_common():
    return {'train_iters': 50_000, 'n_points': n_out, 'seed': new_seed()}

label_match_experiment = functools.partial(experiment, task_class=LabelRingMatch)

all_cases = []
for _ in range(n_iters):
    all_cases.extend([
        Case('MLP', MlpConfig(n_out=n_out, n_layers=3, n_hidden=512), label_match_experiment, experiment_args=make_common()),
        Case('Transformer', TransformerConfig(n_out=n_out, n_layers=3, n_hidden=512, use_mlp_layers=True, pos_emb=True), label_match_experiment, experiment_args=make_common()),
        Case('MNN', PolyConfig(n_out=n_out, n_layers=1, n_hidden=512), label_match_experiment, experiment_args=make_common()),
    ])

for case in tqdm(all_cases):
    print('CASE', case.name)
    print('seed', case.experiment_args['seed'])
    case.run()

## TODO: incorporate utility to plot training curves

# <codecell>
### RADIUS GENERALIZATION
radii = [0.1, 0.5, 1, 2, 4, 8]
for r in radii:
    eval_cases(all_cases, 
               [LabelRingMatch(n_points=n_out, radius=r, batch_size=1024, seed=c.experiment_args['seed']) for c in all_cases], 
               key_name=f'rad_{r}')


# <codecell>
df = pd.DataFrame(all_cases)
df_rad = pd.DataFrame(df['info'].tolist())


# <codecell>
df.to_pickle('tmp.pkl')

# <codecell>
df = pd.read_pickle('tmp.pkl')

# <codecell>
df_rad = pd.DataFrame(df['info'].tolist())
df_rad = df_rad.rename(lambda x: x.split('_')[1], axis='columns')

plot_df = df[['name']].join(df_rad) \
                      .melt(id_vars='name', var_name='radius', value_name='acc')
plot_df['acc'] = plot_df['acc'].astype('float32')

# <codecell>
plt.gcf().set_size_inches(6, 3)
sns.barplot(plot_df, x='radius', y='acc', hue='name')

plt.tight_layout()
plt.savefig('fig/match_label_radius.png')

# <codecell>

'''
Tests to try:
- for large n_out, can a model handle small n_out? (with same number of classes)
- vary depth and compare to MNN

- eventually: probe MLP for evidence of "multiplicative" interactions
'''


# <codecell>
n_choices = 6
task = RingMatch(n_points=n_choices, scramble=False)

# config = MlpConfig(n_out=n_choices, n_layers=3, n_hidden=256)
# config = DotMlpConfig(n_out=n_choices, use_initial_proj=False, last_token_only=True)
# config = PolyConfig(n_out=n_choices, n_layers=1, n_hidden=512)

config = TransformerConfig(n_out=n_choices, n_hidden=64, n_layers=2, n_mlp_layers=2, pos_emb=True)

state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=40_000, lr=1e-4, l1_weight=1e-4)

# %%
import jax.numpy as jnp
task_scramble = RingMatch(n_points=n_choices, scramble=True, batch_size=1024)
task_large = RingMatch(n_points=n_choices, radius=1, batch_size=1024)

xs, labs = next(task_scramble)

# NOTE: the tailored bottleneck model learns awfully
state.params['Dense_0']['bias'] = jnp.zeros(6)
ker = jnp.eye(6)
ker = ker.at[-1, -1].set(0)
state.params['Dense_0']['kernel'] = ker
state.params['Dense_0']['kernel']
# <codecell>
logits = state.apply_fn({'params': state.params}, xs)
preds = logits.argmax(axis=1)
eval_acc = np.mean(labs == preds)
eval_acc

# <codecell>
task = RingMatch(n_points=n_choices, scramble=True)

xs, ys = next(task)
x = xs[0]
a = np.einsum('jh,ih->ji', x, x)
plt.plot(a[-1][:5])
print(np.argmax(a[-1,:-1]))
ys[0]

# <codecell>
import flax.linen as nn

m = config.to_model()
out = nn.tabulate(m, jax.random.PRNGKey(1))(np.ones((20, 6, 2)))
print(out)

# jax.tree_map(lambda x: x.shape, state.params)