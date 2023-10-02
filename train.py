"""
Model definitions
"""

# <codecell>
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import struct
from flax.training import train_state

from task.ti import TiTask

from model.mlp import MlpConfig
from model.transformer import TransformerConfig


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


def create_train_state(rng, model, lr=1e-4, **opt_kwargs):
    params = model.init(rng, jnp.ones((1,2), dtype=jnp.int32))['params']
    tx = optax.adamw(learning_rate=lr, **opt_kwargs)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty()
    )


@jax.jit
def train_step(state, batch):
    x, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels)
        assert len(loss.shape) == 1
        return loss.sum()
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(state, batch):
    x, labels = batch
    logits = state.apply_fn({'params': state.params}, x)
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()

    preds = logits > 0
    acc = jnp.mean(preds == labels)

    metrics = Metrics(accuracy=acc, loss=loss)
    metrics = state.metrics.merge(metrics)
    state = state.replace(metrics=metrics)
    return state


def train(config, data_iter, train_iters=1_000, test_iters=100, test_every=100, seed=None):
    if seed is None:
        seed = new_seed()
    
    init_rng = jax.random.key(seed)
    model = config.to_model()
    state = create_train_state(init_rng, model)

    hist = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for step, batch in zip(range(train_iters), data_iter):
        state = train_step(state, batch)
        state = compute_metrics(state, batch)

        if (step + 1) % test_every == 0:
            hist['train_loss'].append(state.metrics.loss)
            hist['train_acc'].append(state.metrics.accuracy)

            state = state.replace(metrics=Metrics.empty())
            test_state = state
            for _, test_batch in zip(range(test_iters), data_iter):
                test_state = compute_metrics(test_state, test_batch)
            
            hist['test_loss'].append(test_state.metrics.loss)
            hist['test_acc'].append(test_state.metrics.accuracy)

            _print_status(step+1, hist)
    
    return state, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  loss={hist["test_loss"][-1]:.4f}   acc={hist["test_acc"][-1]:.4f}')


if __name__ == '__main__':
    # <codecell>
    vocab_size = 5
    task = TiTask(dist=[1, 2])

    config = TransformerConfig(vocab_size=vocab_size, n_layers=2, pos_emb=False)
    state, hist = train(config, data_iter=iter(task))

    # <codecell>
    logits = [state.apply_fn({'params': state.params}, jnp.array([[0, i]]).astype(jnp.int32)) for i in range(1, vocab_size)]

    # %%
    plt.plot(logits)

