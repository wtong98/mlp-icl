"""
Comparing behavior of models versus KNN (on finite data sizes), at the
accuracy and logit level on the match task.

Observations:
- At a high level, KNN accuracy tracks that of other models
- Probability outputs from the KNN seems to match that of trained MLPs, especially
for shallow MLPs trained to a plateau
- KNN accuracy tends to match when training accuracy hits ~90% on match task <-- TODO: check with benchmark plots

Things to solidify:
- Hypothesis: KNN matched at point of perfect training accuracy or a plateau
- Measure divergence for deeper models / alternative models
- Compare to other tasks (e.g. Oddball)
"""

# <codecell>
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import *

import sys
sys.path.append('../')
from train import train
from model.knn import KnnConfig
from model.mlp import MlpConfig
from task.match import RingMatch


# <codecell>
df = pd.read_pickle('remote/6_knn_comparison/res.pkl')

def extract_plot_vals(row):
    data_size = row['data_size']
    if np.isnan(data_size):
        data_size = row['train_task'].data_size

    return pd.Series([
        row['name'],
        int(data_size),
        row['info']['eval_acc'].item(),
    ], index=['name', 'data_size', 'acc'])

plot_df = df.apply(extract_plot_vals, axis=1)

# <codecell>
### PLOT LOSS FIGS
# NOTE: not that helpful, see fine-grained plateau plots below

# fig_dir = Path('fig/match_knn_loss')
# if not fig_dir.exists():
#     fig_dir.mkdir()

# for i, row in df.iterrows():
#     if row['name'] == 'KNN':
#         continue

#     acc = [m['accuracy'] for m in row['hist']['train']]

#     plt.plot(acc)
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title(row['name'])

#     plt.tight_layout()
#     plt.savefig(fig_dir / f'{row["name"]}_{i}.png')
#     plt.show()

# <codecell>
plt.gcf().set_size_inches(10, 3.5)
g = sns.barplot(plot_df, x='data_size', y='acc', hue='name')
g.legend_.set_title(None)

plt.tight_layout()
plt.savefig('fig/match_knn_gen.png')

# <codecell>
### EXAMINE LOGITS MATCH
n_out = 6
n_models = 5

task = RingMatch(data_size=32, n_points=n_out)

all_states = []

for _ in range(n_models):
    model = MlpConfig(n_layers=1, n_hidden=256, n_out=n_out)
    state, hist = train(model, task, loss='ce', train_iters=3_000, test_every=1_000)

    all_states.append(state)

# <codecell>
full_task = RingMatch(n_points=n_out, batch_size=1024)
xs, ys = next(full_task)
all_mlp_probs = []

for state in all_states:
    logits = state.apply_fn({'params': state.params}, xs)
    mlp_probs = jax.nn.softmax(logits, axis=-1)
    mlp_acc = np.mean(logits.argmax(-1) == ys)
    print('MLP', mlp_acc)

    all_mlp_probs.append(mlp_probs)

data = task.data[0].reshape(task.data_size, -1)
labs = task.data[1]
knn = KnnConfig(beta=3, n_classes=n_out, xs=data, ys=labs).to_model()

xs_knn = xs.reshape(1024, -1)
knn_probs = knn(xs_knn)
knn_preds = knn_probs.argmax(-1)

knn_acc = np.mean(knn_preds == ys)

print('KNN', knn_acc)

# <codecell>
fig_dir = Path('fig/match_knn_example_probs')
if not fig_dir.exists():
    fig_dir.mkdir()

for idx in range(25):
    for i, prob in enumerate(all_mlp_probs):
        if i == 0:
            plt.plot(prob[idx], 'o--', alpha=0.4, color='C0', label='MLP')
        else:
            plt.plot(prob[idx], 'o--', alpha=0.4, color='C0')

    plt.plot(knn_probs[idx], 'o--', color='C1', label='near-neigh')
    plt.xlabel('Choice')
    plt.ylabel('Probability')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / f'{idx}.png')
    plt.show()


# <codecell>
### PLOTTING PLATEAUS
n_iters = 3
n_out = 6
data_sizes = [16, 32, 64, 128]
train_iters=10_000

all_cases = []
for _ in range(n_iters):
    for size in data_sizes:

        common_seed = new_seed()
        common_task_args = {'n_points': n_out, 'seed': common_seed}
        common_train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 10}

        curr_tasks = [
            Case('MLP', MlpConfig(n_out=n_out, n_layers=1, n_hidden=256), train_args=common_train_args),
            KnnCase('KNN', KnnConfig(beta=3, n_classes=n_out), task_class=RingMatch, data_size=size, seed=common_seed)
        ]

        for case in curr_tasks:
            case.train_task = RingMatch(data_size=size, **common_task_args)
            case.test_task = RingMatch(batch_size=1024, **common_task_args)

        all_cases.extend(curr_tasks)


for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

task = RingMatch(n_points=n_out, batch_size=1024)
eval_cases(all_cases, task)


# <codecell>
fig_dir = summon_dir('fig/match_knn_example_loss')

for idx in range(0, len(all_cases), 2):
    train_acc = [m.accuracy for m in all_cases[idx].hist['train']]
    test_acc = [m.accuracy for m in all_cases[idx].hist['test']]

    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.axhline(y=all_cases[idx+1].info['eval_acc'], color='purple', label='near-neigh acc')
    plt.title(f'Data size = {all_cases[idx].train_task.data_size}')

    plt.xscale('log')
    plt.xlabel('Step (x10)')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / f'{idx}.png')
    plt.show()