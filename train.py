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

from task.oddball import FreeOddballTask, LineOddballTask
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
    l1_loss: float
    count: int = 0

    @staticmethod
    def empty():
        return Metrics(accuracy=-1, loss=-1, l1_loss=-1)
    
    def merge(self, other):
        total = self.count + 1
        acc = (self.count / total) * self.accuracy + (1 / total) * other.accuracy
        loss = (self.count / total) * self.loss + (1 / total) * other.loss
        l1_loss = (self.count / total) * self.l1_loss + (1 / total) * other.l1_loss
        return Metrics(acc, loss, l1_loss, count=total)


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
    elif loss == 'ce':
        loss_func = optax.softmax_cross_entropy_with_integer_labels
    elif loss == 'mse':
        loss_func = optax.l2_loss
    else:
        raise ValueError(f'unrecognized loss name: {loss}')
    return loss_func

# TODO: more robustly signal need for L1 loss
def l1_loss(params):
    # sum_params = jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), jax.tree_util.tree_leaves(params))
    # return jnp.sum(jnp.array(sum_params))
    loss = 0
    for name in params:
        if 'MBlock' in name:
            z_weights = params[name]['DenseMultiply']['kernel']
            loss += jnp.sum(jnp.abs(z_weights))

    return loss

@partial(jax.jit, static_argnames=('loss',))
def train_step(state, batch, loss='bce', l1_weight=0):
    x, labels = batch
    loss_func = parse_loss_name(loss)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = loss_func(logits, labels)
        assert len(loss.shape) == 1

        l1_term = l1_weight * l1_loss(params)
        return loss.mean() + l1_term
    
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
    l1 = l1_loss(state.params)

    if len(logits.shape) == 1:
        preds = logits > 0
    else:
        preds = logits.argmax(axis=1)
    
    acc = jnp.mean(preds == labels)

    metrics = Metrics(accuracy=acc, loss=loss, l1_loss=l1)
    metrics = state.metrics.merge(metrics)
    state = state.replace(metrics=metrics)
    return state


def train(config, data_iter, loss='bce', train_iters=1_000, test_iters=100, test_every=100, seed=None, l1_weight=0, **opt_kwargs):
    if seed is None:
        seed = new_seed()
    
    init_rng = jax.random.key(seed)
    model = config.to_model()

    samp_x, _ = next(data_iter)
    state = create_train_state(init_rng, model, samp_x, **opt_kwargs)

    hist = {
        'train': [],
        'test': []
    }

    for step, batch in zip(range(train_iters), data_iter):
        state = train_step(state, batch, loss=loss, l1_weight=l1_weight)
        state = compute_metrics(state, batch, loss=loss)

        if (step + 1) % test_every == 0:
            hist['train'].append(state.metrics)

            state = state.replace(metrics=Metrics.empty())
            test_state = state
            for _, test_batch in zip(range(test_iters), data_iter):
                test_state = compute_metrics(test_state, test_batch, loss=loss)
            
            hist['test'].append(test_state.metrics)

            _print_status(step+1, hist)
    
    return state, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  loss={hist["test"][-1].loss:.4f}   l1_loss={hist["test"][-1].l1_loss:.4f}  acc={hist["test"][-1].accuracy:.4f}')


if __name__ == '__main__':
    # domain = -3, 3
    # task = DotProductTask(domain, n_dims=5, n_args=3, batch_size=256)
    # task = FreeOddballTask()
    n_choices = 12
    task = LineOddballTask(n_choices=n_choices, linear_dist=10)

    # config = TransformerConfig(pure_linear_self_att=True)
    # config = TransformerConfig(pos_emb=True, n_emb=None, n_out=6, n_layers=3, n_hidden=128, use_mlp_layers=True, pure_linear_self_att=False)
    config = MlpConfig(n_out=n_choices, n_layers=3, n_hidden=128)

    # config = PolyConfig(n_hidden=128, n_layers=1, n_out=n_choices)
    state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=50_000, lr=1e-4, l1_weight=1e-4)

    # <codecell>
    loss = [m.loss for m in hist['test']]
    l1_loss = [m.l1_loss for m in hist['test']]
    p1 = plt.plot(loss, label='Data loss')[0]
    plt.yscale('log')
    plt.xlabel('Time (x1000 batches)')

    ax = plt.gca().twinx()
    p2 = ax.plot(l1_loss, color='C1', label='L1 loss')[0]

    ax.set_yscale('log')

    plt.legend(handles=[p1, p2], loc='center right')

    ax = plt.gca()
    ax.spines['left'].set_color('C0')
    ax.spines['right'].set_color('C1')
    
    plt.tight_layout()
    # plt.savefig('experiment/fig/loss.png')


    # %%
    state.apply_fn({'params': state.params}, jnp.array([[0.5, 0.5]]), mutable='intm')

    # %%
    x = np.linspace(-4, 4, 50)
    xs = np.stack((x, -x), axis=-1)

    out = state.apply_fn({'params': state.params}, xs)

    plt.plot(x, -x**2)
    plt.plot(x, out, alpha=0.9, linestyle='dashed')

    # <codecell>



