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
        return PolyNet(self)


class PolyNet(nn.Module):

    config: PolyConfig

    def m_block(self, x, n_features=None):
        if n_features is None:
            n_features = self.config.n_hidden

        z = jnp.log(jnp.abs(x))
        z = nn.Dense(n_features, name='DenseMult')(z)
        z = jnp.exp(z)

        beta = 10 # NOTE: sharpen tanh, can make trainable
        s = nn.tanh(beta * x)
        s_prod = jnp.einsum('ij,ik->ijk', s, s).reshape(s.shape[0], -1)

        s = jnp.concatenate((s, s_prod), axis=-1)
        s = nn.Dense(n_features, name='SignOut')(s)
        s = nn.tanh(s)

        out = s * z
        return out
    
    def poly_block(self, x, n_features=None):
        if n_features is None:
            n_features = self.config.n_hidden

        x = self.m_block(x, n_features=self.config.n_hidden)
        x = nn.Dense(n_features)(x)
        return x

    def _fwd_product_sep_sign(self, x):
        return self.m_block(x, n_features=1).flatten()
    
    def _fwd_product_sep_sign_full(self, x):
        return self.poly_block(x, n_features=1).flatten()

    @nn.compact
    def __call__(self, x):
        if self.config.vocab_size is not None:
            x = nn.Embed(
                num_embeddings=self.config.vocab_size,
                features=self.config.n_emb)(x)
        
        x = x.reshape(x.shape[0], -1)

        return self._fwd_product_sep_sign_full(x)


if __name__ == '__main__':
    model = PolyConfig(10).to_model()
    print(model.tabulate(jax.random.key(10), jnp.ones((32, 2)).astype(jnp.int32)))
    # params = model.init(jax.random.key(0), jnp.ones((32, 2)).astype(jnp.int32))
    # print(jax.tree_map(lambda x: x.shape, params))