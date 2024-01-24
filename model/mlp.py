"""
Simple MLP model
"""
import jax
import numpy as np
from flax import linen as nn, struct


@struct.dataclass
class MlpConfig:
    """Global hyperparamters"""
    vocab_size: int | None = None
    n_layers: int = 2
    n_emb: int = 64
    n_hidden: int = 128
    n_out: int = 1
    act_fn: str = jax.nn.relu

    def to_model(self):
        return MLP(self)


class MLP(nn.Module):

    config: MlpConfig

    @nn.compact
    def __call__(self, x):
        if self.config.vocab_size is not None:
            x = nn.Embed(
                num_embeddings=self.config.vocab_size,
                features=self.config.n_emb)(x)
        
        x = x.reshape(x.shape[0], -1)

        for _ in range(self.config.n_layers):
            x = nn.Dense(self.config.n_hidden)(x)
            x = self.config.act_fn(x)
    
        out = nn.Dense(self.config.n_out)(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out


@struct.dataclass
class RfConfig:
    """Global hyperparamters"""
    n_in: int
    n_hidden: int = 128
    n_out: int = 1
    scale: float = 1
    use_quadratic_activation: bool = True

    def to_model(self):
        return RF(self)


class RF(nn.Module):

    config: RfConfig

    def setup(self):
        scale = self.config.scale / np.sqrt(self.config.n_hidden)
        self.w_rf1 = scale * np.random.randn(self.config.n_in, self.config.n_hidden)
        self.w_rf2 = scale * np.random.randn(self.config.n_hidden, self.config.n_hidden)

        self.readout = nn.Dense(self.config.n_out)

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = x @ self.w_rf1
        # x = nn.relu(x)
        x = x @ self.w_rf2

        if self.config.use_quadratic_activation:
            x = x**2

        out = self.readout(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out