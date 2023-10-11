"""
Simple MLP model
"""

# <codecell>

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn, struct


@struct.dataclass
class PolyConfig:
    """Global hyperparamters"""
    vocab_size: int | None = None
    n_layers: int = 2
    n_emb: int = 64
    n_hidden: int = 128

    def to_model(self):
        if self.n_hidden % 2 != 0:
            raise ValueError(f'n_hidden should be even, got n_hidden={self.n_hidden}')
        return PolyNet(self)


class PolyNet(nn.Module):

    config: PolyConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_emb)(x)
        
        x = x.reshape(x.shape[0], -1)

        proj_dim = self.config.n_hidden // 2
        for _ in range(self.config.n_layers):
            x = nn.Dense(proj_dim)(x)

            z = jnp.log(x)
            z = nn.Dense(proj_dim)(z)
            z = jnp.exp(x)

            x = jnp.concatenate((x, z), axis=-1)
    
        out = nn.Dense(1)(x).flatten()
        return out


if __name__ == '__main__':
    model = PolyConfig(10).to_model()
    print(model.tabulate(jax.random.key(10), jnp.ones((32, 2)).astype(jnp.int32)))
    # params = model.init(jax.random.key(0), jnp.ones((32, 2)).astype(jnp.int32))
    # print(jax.tree_map(lambda x: x.shape, params))