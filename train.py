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
from task.match import RingMatch, LabelRingMatch
from task.oddball import FreeOddballTask, LineOddballTask
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


def create_train_state(rng, model, dummy_input, lr=1e-4, optim=optax.adamw, **opt_kwargs):
    params = model.init(rng, dummy_input)['params']
    tx = optim(learning_rate=lr, **opt_kwargs)
    # tx = optax.sgd(learning_rate=lr, **opt_kwargs)

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
        loss_func = optax.squared_error
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

@partial(jax.jit, static_argnames=('gamma', 'loss',))
def train_step(state, batch, gamma=None, loss='bce', l1_weight=0):
    x, labels = batch
    loss_func = parse_loss_name(loss)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        train_loss = loss_func(logits, labels)

        # print("EARLY LOGITS", jnp.max(logits).item())
        # p1 = jax.nn.log_sigmoid(logits)
        # p2 = jax.nn.log_sigmoid(-logits)
        # train_loss = -labels * p1 - (1 - labels) * p2
        
        # print("P1", p1.min().item())
        # print("P2", p2.min().item())
        # print('EARLY LOSS', train_loss.mean().item())

        if loss == 'bce' and len(labels.shape) > 1:
            assert logits.shape == train_loss.shape
            train_loss = train_loss.mean(axis=-1)

        assert len(train_loss.shape) == 1

        l1_term = l1_weight * l1_loss(params)

        # TODO: numerically unstable
        if gamma is not None:
            logits = jax.lax.stop_gradient(logits)
            log_fac = (1 - labels) * logits * ((1 / gamma) - 1) \
                        + jax.nn.softplus(logits) \
                        - jax.nn.softplus((1 / gamma) * logits)
            train_loss = jnp.exp(log_fac + jnp.log(train_loss))

            # print('LOGITS', logits.max().item())
            # print('SOFT+', jax.nn.softplus(logits).max().item())
            # print('LOG_FAC', jnp.sum(log_fac).item())
            # print('LOG LOSS', jnp.sum(jnp.log(train_loss)).item())
            # print('SUM', jnp.max(log_fac + jnp.log(train_loss)).item())
            # print('LOSS', train_loss.max().item())
            # print('---')

        return train_loss.mean() + l1_term
    
    grads = jax.grad(loss_fn)(state.params)
    # if gamma is not None:
    #     logits = state.apply_fn({'params': state.params}, x)
    #     avg_logit = jnp.mean(jnp.abs(logits))
    #     scale_factor = (1 + jnp.exp(avg_logit)) / (1 + jnp.exp((1 / gamma) * avg_logit))
    #     # scale_factor = jnp.exp(avg_logit)
    #     # scale_factor = jnp.clip(scale_factor, a_min=None, a_max=1e20)
    #     grads = jax.tree.map(lambda g: scale_factor * g, grads)

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
    
    if len(labels.shape) > 1:
        labels = labels.argmax(axis=1)
    
    acc = jnp.mean(preds == labels)

    metrics = Metrics(accuracy=acc, loss=loss, l1_loss=l1)
    metrics = state.metrics.merge(metrics)
    state = state.replace(metrics=metrics)
    return state


def train(config, data_iter, 
          test_iter=None, 
          loss='ce', gamma=None,
          train_iters=10_000, test_iters=100, test_every=1_000, save_params=False,
          early_stop_n=None, early_stop_key='loss', early_stop_decision='min' ,
          optim=optax.adamw,
          seed=None, 
          l1_weight=0, **opt_kwargs):
    if seed is None:
        seed = new_seed()
    
    if test_iter is None:
        test_iter = data_iter
    
    init_rng = jax.random.key(seed)
    model = config.to_model()

    samp_x, _ = next(data_iter)
    state = create_train_state(init_rng, model, samp_x, optim=optim, **opt_kwargs)

    hist = {
        'train': [],
        'test': [],
        'params': []
    }

    for step, batch in zip(range(train_iters), data_iter):
        state = train_step(state, batch, loss=loss, l1_weight=l1_weight, gamma=gamma)
        state = compute_metrics(state, batch, loss=loss)

        if ((step + 1) % test_every == 0) or ((step + 1) == train_iters):
            hist['train'].append(state.metrics)

            state = state.replace(metrics=Metrics.empty())
            test_state = state
            for _, test_batch in zip(range(test_iters), test_iter):
                test_state = compute_metrics(test_state, test_batch, loss=loss)
            
            hist['test'].append(test_state.metrics)

            _print_status(step+1, hist)

            if save_params:
                hist['params'].append(state.params)
        
            if early_stop_n is not None and len(hist['train']) > early_stop_n:
                last_n_metrics = np.array([getattr(m, early_stop_key) for m in hist['train'][-early_stop_n - 1:]])
                if early_stop_decision == 'min' and np.all(last_n_metrics[0] < last_n_metrics[1:]) \
                or early_stop_decision == 'max' and np.all(last_n_metrics[0] > last_n_metrics[1:]):
                    print(f'info: stopping early with {early_stop_key} =', last_n_metrics[-1])
                    break
    
    return state, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  train_loss={hist["train"][-1].loss:.4f}   train_acc={hist["train"][-1].accuracy:.4f}   test_acc={hist["test"][-1].accuracy:.4f}')


if __name__ == '__main__':
    # domain = -3, 3
    # task = DotProductTask(domain, n_dims=5, n_args=3, batch_size=256)
    n_choices = 6
    # task = FreeOddballTask(n_choices=n_choices, one_hot=True)
    # task = LineOddballTask(n_choices=n_choices, linear_dist=10)
    task = RingMatch(n_points=n_choices)

    # config = TransformerConfig(pos_emb=True, n_emb=None, n_out=6, n_layers=3, n_hidden=128, use_mlp_layers=True, pure_linear_self_att=False)
    config = MlpConfig(n_out=n_choices, n_layers=3, n_hidden=512)

    # config = PolyConfig(n_hidden=512, n_layers=1, n_out=n_choices, disable_signage=False)
    state, hist = train(config, data_iter=iter(task), loss='ce', test_every=1000, train_iters=50_000, early_stop_n=3, early_stop_decision='max', early_stop_key='accuracy', lr=1e-4, l1_weight=1e-4)

    """Observations: transformer performs handsomely, with great sample efficiency. Then
    MLP, then MNN performs the worst (but still reasonably well)""" # TODO: solidify <-- STOPPED HERE

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



