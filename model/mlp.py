"""
Simple MLP model
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn, struct
from flax.training import train_state


@struct.dataclass
class MlpConfig:
    """Global hyperparamters"""
    vocab_size: int | None = None
    n_layers: int = 2
    n_emb: int = 64
    n_hidden: int = 128

    def to_model(self):
        return MLP(self)


class MLP(nn.Module):

    config: MlpConfig

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