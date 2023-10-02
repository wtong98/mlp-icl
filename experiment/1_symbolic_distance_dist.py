"""
Symbolic Distance

This experiment measures the distribution of symbolic distance in MLPs trained
under different variants of the TI task. It appears that models exposed to
adjacencies >1 quickly exhibit a biologically-realistic symbolic distance
effect that is absence in the =1 case.
"""

# <codecell>
import sys

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.append('../')
from train import train
from model.mlp import MlpConfig
from task.ti import TiTask

def run_exp(vocab_size=5, fig_path=None):
    n_replicas = 10
    train_iters = 1_000
    test_every = 500

    all_results = []
    for max_dist in range(1, vocab_size):
        print('PROCESSING UP TO DIST', max_dist)
        dists = np.arange(1, max_dist+1)
        task = TiTask(n_items=vocab_size, dist=dists)

        all_logits = []
        for _ in range(n_replicas):
            config = MlpConfig(vocab_size=vocab_size)
            state, _ = train(config, data_iter=iter(task), train_iters=train_iters, test_every=test_every)
            logits = [state.apply_fn({'params': state.params}, jnp.array([[0, i]]).astype(jnp.int32)) for i in range(1, vocab_size)]
            all_logits.append(logits)
        
        all_results.append(all_logits)

    all_results = np.array(all_results).squeeze()
    print('done!')

    fig, axs = plt.subplots(1, vocab_size-1, figsize=(2 * (vocab_size-1), 2))
    for max_dist, res, ax in zip(range(1, vocab_size), all_results, axs):
        for line in res:
            ax.plot(np.arange(1, vocab_size), line, color=f'C0', alpha=0.7)
            ax.set_title(f'Adjacency = {max_dist}')
            ax.set_xlabel('Symbol distance')
            ax.set_ylabel('Logit')

    fig.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)


# <codecell>
run_exp(vocab_size=5, fig_path='fig/symb_dist_5.png')

# <codecell>
run_exp(vocab_size=10, fig_path='fig/symb_dist_10.png')
# %%

vocab_size = 5
train_iters = 1_000
test_every = 100
full_task = TiTask(dist=[1,2,3,4])
p = full_task.all_pairs


task = TiTask(dist=[1,])

config = MlpConfig(vocab_size=vocab_size, train_iters=train_iters, test_every=test_every)
state, _ = train(config, data_iter=iter(task))

# <codecell>
logits = [state.apply_fn({'params': state.params}, jnp.array(p[i].T).astype(jnp.int32)) for i in range(1, vocab_size)]
fig, axs = plt.subplots(1, 4, figsize=(8, 2))
for logit, ax in zip(logits, axs):
    ax.plot(np.arange(len(logit)), logit, '--o')
    ax.set_ylim(4, 12)

fig.tight_layout()




# %%
