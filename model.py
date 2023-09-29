"""
Model definitions
"""

# <codecell>
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn, struct
from flax.training import train_state

from task.ti import TiTask


def new_seed(): return np.random.randint(1, np.iinfo(np.int32).max)


@struct.dataclass
class ModelConfig:
    """Global hyperparamters"""
    vocab_size: int | None = None
    n_layers: int = 2
    n_emb: int = 64
    n_hidden: int = 128

    train_iters: int = 1_000
    test_iters: int = 100
    test_every: int = 100


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
    params = model.init(rng, jnp.ones(2, dtype=jnp.int32))['params']
    tx = optax.adamw(learning_rate=lr, **opt_kwargs)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty()
    )


class FFNN(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_emb)(x)
        
        x = x.reshape(-1, 2 * self.config.n_emb)

        for _ in range(self.config.n_layers):
            x = nn.Dense(self.config.n_hidden)(x)
            x = nn.relu(x)
    
        out = nn.Dense(1)(x).flatten()
        return out


@jax.jit
def train_step(state, batch):
    x, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels).sum()
        return loss
    
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


def train(config: ModelConfig, data_iter, seed=None):
    if seed is None:
        seed = new_seed()
    
    init_rng = jax.random.key(seed)
    model = FFNN(config)
    state = create_train_state(init_rng, model)

    hist = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for step, batch in zip(range(config.train_iters), data_iter):
        state = train_step(state, batch)
        state = compute_metrics(state, batch)

        if (step + 1) % config.test_every == 0:
            hist['train_loss'].append(state.metrics.loss)
            hist['train_acc'].append(state.metrics.accuracy)

            state = state.replace(metrics=Metrics.empty())
            test_state = state
            for _, test_batch in zip(range(config.test_iters), data_iter):
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
    task = TiTask(dist=[1,2,3,4])

    config = ModelConfig(vocab_size=vocab_size, train_iters=1_000, test_every=100)
    state, hist = train(config, data_iter=iter(task))

    # <codecell>
    logits = [state.apply_fn({'params': state.params}, jnp.array([[0, i]]).astype(jnp.int32)) for i in range(1, vocab_size)]

    # %%
    plt.plot(logits)

