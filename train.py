"""
Model definitions
"""

# <codecell>
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import struct
from flax.training import train_state

from task.function import MultiplicationTask, DotProductTask
from task.ti import TiTask

from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from model.poly import PolyConfig


def new_seed(): return np.random.randint(1, np.iinfo(np.int32).max)


@struct.dataclass
class Metrics:
    accuracy: float
    loss: float
    count: int = 0

    @staticmethod
    def empty():
        return Metrics(accuracy=-1, loss=-1)
    
    def merge(self, other):
        total = self.count + 1
        acc = (self.count / total) * self.accuracy + (1 / total) * other.accuracy
        loss = (self.count / total) * self.loss + (1 / total) * other.loss
        return Metrics(acc, loss, count=total)


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(rng, model, dummy_input, lr=1e-4, **opt_kwargs):
    params = model.init(rng, dummy_input)['params']
    tx = optax.adamw(learning_rate=lr, **opt_kwargs)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty()
    )

def parse_loss_name(loss):
    loss_func = None
    if loss == 'bce':
        loss_func = optax.sigmoid_binary_cross_entropy
    elif loss == 'mse':
        loss_func = optax.l2_loss
    else:
        raise ValueError(f'unrecognized loss name: {loss}')
    return loss_func


@partial(jax.jit, static_argnames=('loss',))
def train_step(state, batch, loss='bce'):
    x, labels = batch
    loss_func = parse_loss_name(loss)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = loss_func(logits, labels)
        assert len(loss.shape) == 1
        # print('LOSS', loss)
        return loss.mean()
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@partial(jax.jit, static_argnames=('loss',))
def compute_metrics(state, batch, loss='bce'):
    x, labels = batch
    logits = state.apply_fn({'params': state.params}, x)
    loss_func=parse_loss_name(loss)
    loss = loss_func(logits, labels).mean()

    preds = logits > 0
    acc = jnp.mean(preds == labels)

    metrics = Metrics(accuracy=acc, loss=loss)
    metrics = state.metrics.merge(metrics)
    state = state.replace(metrics=metrics)
    return state


def train(config, data_iter, loss='bce', train_iters=1_000, test_iters=100, test_every=100, seed=None):
    if seed is None:
        seed = new_seed()
    
    init_rng = jax.random.key(seed)
    model = config.to_model()

    samp_x, _ = next(data_iter)
    state = create_train_state(init_rng, model, samp_x)

    hist = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for step, batch in zip(range(train_iters), data_iter):
        state = train_step(state, batch, loss=loss)
        state = compute_metrics(state, batch, loss=loss)

        if (step + 1) % test_every == 0:
            hist['train_loss'].append(state.metrics.loss)
            hist['train_acc'].append(state.metrics.accuracy)

            state = state.replace(metrics=Metrics.empty())
            test_state = state
            for _, test_batch in zip(range(test_iters), data_iter):
                test_state = compute_metrics(test_state, test_batch, loss=loss)
            
            hist['test_loss'].append(test_state.metrics.loss)
            hist['test_acc'].append(test_state.metrics.accuracy)

            _print_status(step+1, hist)
    
    return state, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  loss={hist["test_loss"][-1]:.4f}   acc={hist["test_acc"][-1]:.4f}')


if __name__ == '__main__':
    domain = -3, 3
    task = MultiplicationTask(domain)
    # task = TiTask(dist=[1,2,3])

    config = PolyConfig(n_hidden=2, n_layers=1)
    state, hist = train(config, data_iter=iter(task), loss='mse', test_every=1000, train_iters=50_000)


    # %%
    state.apply_fn({'params': state.params}, jnp.array([[0.5, 0.5]]), mutable='intm')

    # %%
    x = np.linspace(-4, 4, 50)
    xs = np.stack((x, x), axis=-1)

    out = state.apply_fn({'params': state.params}, xs)

    plt.plot(x, x**2)
    plt.plot(x, out)


# <codecell>
x1 = 1
x2 = 2

w1 = state.params['Dense_0']['kernel']
b1 = state.params['Dense_0']['bias']

z1 = state.params['z_kernel_0']
zb1 = state.params['z_bias_0']

# wo = state.params['Dense_1']['kernel']
# bo = state.params['Dense_1']['bias']

h1 = x1 * w1[0] + x2 * w1[1] + b1
sign = np.cos(np.pi * np.sum(z1))
h2 = np.exp(np.log(x1) * z1[0] + np.log(x2) * z1[1] + zb1) * sign

# out = h1 * wo[0] + h2 * wo[1] + bo
# out
h2



    # vocab_size=5
    # logits = [state.apply_fn({'params': state.params}, jnp.array([[0, i]]).astype(jnp.int32)) for i in range(1, vocab_size)]
    # plt.plot(logits)

    # %%


# %%
state.params
# %%
