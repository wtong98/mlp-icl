"""Alex's mup experiment: will same-different work?"""

# <codecell>
from collections import defaultdict

import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.random as random
import numpy as np
import optax


def batch_choice(a, n_elem, batch_size, rng=None):
    assert n_elem <= len(a), f'require n_elem <= len(a), got n_elem={n_elem} and len(a)={len(a)}'

    if rng is None:
        rng = np.random.default_rng(None)

    idxs = np.tile(a, (batch_size, 1))
    idxs = rng.permuted(idxs, axis=1)
    idxs = idxs[:,:n_elem]
    return idxs


class SameDifferent:
    def __init__(self, n_symbols=None, task='hard',
                 n_dims=2, thresh=0, radius=1,    # soft/hard params
                 n_seen=None, sample_seen=True,   # token params
                 seed=None, reset_rng_for_data=True, batch_size=128) -> None:

        if task == 'token':
            assert n_symbols is not None and n_symbols >= 4, 'if task=token, n_symbols should be >= 4'
            
            if n_seen is None:
                n_seen = n_symbols // 2

        self.n_symbols = n_symbols
        self.task = task
        self.n_dims = n_dims
        self.thresh = thresh
        self.radius = radius
        self.n_seen = n_seen
        self.sample_seen = sample_seen
        self.seed = seed
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        if self.n_symbols is not None:
            self.symbols = self.rng.standard_normal((self.n_symbols, self.n_dims)) / np.sqrt(self.n_dims)

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        if self.task == 'soft':
            return self._sample_soft()
        elif self.task == 'hard':
            return self._sample_hard()
        elif self.task == 'token':
            return self._sample_token()
        else:
            raise ValueError(f'unrecognized task type: {self.task}')

    def _sample_soft(self):
        xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims)) / np.sqrt(self.n_dims)
        # norms = np.linalg.norm(xs, axis=-1, keepdims=True)
        # xs = xs / norms * self.radius

        x0, x1 = xs[:,0], xs[:,1]
        ys = (np.einsum('bi,bi->b', x0, x1) > self.thresh).astype('float')
        return xs, ys.flatten()
    
    def _sample_hard(self):
        if self.n_symbols is None:
            xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims)) / np.sqrt(self.n_dims)
        else:
            sym_idxs = batch_choice(np.arange(self.n_symbols), 2, batch_size=self.batch_size, rng=self.rng)
            xs = self.symbols[sym_idxs]

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys
    
    def _sample_token(self):
        if self.sample_seen:
            xs = batch_choice(np.arange(0, self.n_seen), 2, batch_size=self.batch_size, rng=self.rng)
        else:
            xs = batch_choice(np.arange(self.n_seen, self.n_symbols), 2, batch_size=self.batch_size, rng=self.rng)

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys

    def __iter__(self):
        return self


def init_params(keys, widths):
    params = []
    for i in range(len(widths)-1):
        params += [random.normal(keys[i], (widths[i], widths[i+1]))]
    return params

def apply_fn(params, x, gamma0):
    for i in range(len(params)-2):
        x = jnp.dot(x, params[i]) / jnp.sqrt(x.shape[-1])
        x = nn.relu(x)
    # Note the way I don't divide by the square root at the last layer
    x = jnp.dot(x, params[-1]) / jnp.sqrt(x.shape[-1])
    return x.flatten()/gamma0

def loss_fn(params, X, y, gamma0):
    y_pred = apply_fn(params, X, gamma0)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(y_pred, y))

# Total symbol set size:
P = 8
# Input dimension:
D = 128
# Width:
N = 512
# Batch size
B = 128

task = SameDifferent(n_symbols=P, n_dims=D, batch_size=B)
test_task = SameDifferent(n_symbols=None, n_dims=D, batch_size=1024)
xs, ys = next(task)

init_key, data_key = random.split(random.PRNGKey(0), 2)

# Play around with adjusting these
# gamma0s = [0.1, 0.5, 1, 5, 10, -1]
gamma0s = [1]
gamma_lr = 100
eta0 = 1
T = 100_000
N = 256

final_train_accs = []
final_test_accs = []

losses = defaultdict(list)
for gamma0 in gamma0s:
    print(f"gamma0 = {gamma0}:")
    widths = [2*D, N, N, 1]
    init_keys = random.split(init_key, len(widths))
    params = init_params(init_keys, widths)
    eta = N * eta0
    if gamma0 > 0:
        tx = optax.sgd(5 * eta * gamma0)
    else:
        gamma0 = 1
        tx = optax.adam(1e-2)

    opt_state = tx.init(params)
    # We train for one epoch, where each batch is 'fresh'
    for t in range(T):
        X_batch, y_batch = next(task)
        X_batch = X_batch.reshape((X_batch.shape[0], -1))

        # Comptue gradients
        loss, grad = jax.value_and_grad(loss_fn)(params, X_batch, y_batch, gamma0)
        # Record loss
        losses[N].append(loss)
        # Update the optimizer appropriately
        # updates, opt_state = tx.update(grad, opt_state)
        # params = optax.apply_updates(params, updates)

        eta = eta0 * np.exp((1 - (1 / gamma_lr)) / (5 * loss + 1e-4))
        params = [(w - eta * dw) for w, dw in zip(params, grad)]

        if loss != loss:
            print("Loss is NaN")
            break
        if t % 100 == 0:
            X_train, y_train = next(task)
            X_train = X_train.reshape(X_train.shape[0], -1)
            pred_train = (apply_fn(params, X_train, gamma0) > 0).astype(float)
            acc_train = np.mean(pred_train == y_train)

            X_test, y_test = next(test_task)
            X_test = X_test.reshape(X_test.shape[0], -1)
            pred_test = (apply_fn(params, X_test, gamma0) > 0).astype(float)
            acc_test = np.mean(pred_test == y_test)

            print(f"Loss: {loss:.3f}  train_acc: {acc_train:.3f}   test_acc: {acc_test:.3f}")
        
    final_train_accs.append(acc_train.item())
    final_test_accs.append(acc_test.item())

# <codecell>
import pandas as pd
import seaborn as sns

df = pd.DataFrame({'train_acc': final_train_accs, 'test_acc': final_test_accs, 'gamma0': gamma0s}) \
       .melt(id_vars=['gamma0'], var_name='acc_type', value_name='acc')
df

# <codecell>
g = sns.barplot(df, x='gamma0', y='acc', hue='acc_type')
g.figure.savefig('fig/mup_demo.png')
