"""
Figure for same-different task
"""

# <codecell>
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys


from common import *

fig_dir = Path('fig/final')
set_theme()

# <codecell>
df = collate_dfs('remote/15_same_diff/generalize')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['test_task'].n_vocab,
        row['info']['acc_unseen'].item(),
    ], index=['name', 'n_vocab', 'acc_unseen'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True)
plot_df

# <codecell>
g = sns.lineplot(plot_df, x='n_vocab', y='acc_unseen', marker='o')
g.set_xscale('log', base=2)

g.axhline(y=0.5, linestyle='dashed', color='k', alpha=0.3)
g.set_xlabel(r'$|\mathcal{X}|$')
g.set_ylabel(r'Accuracy on $\mathcal{X}_{uns}$')

fig = g.figure
fig.tight_layout()
fig.savefig(fig_dir / 'same_diff_mlp_acc.png')
plt.show()