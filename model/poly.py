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
        if self.config.vocab_size is not None:
            x = nn.Embed(
                num_embeddings=self.config.vocab_size,
                features=self.config.n_emb)(x)
        
        x = x.reshape(x.shape[0], -1)

        # print('X', x)

        proj_dim = self.config.n_hidden // 2

        for i in range(self.config.n_layers):
            x_lin = nn.Dense(proj_dim)(x)

            # print('X fore log', x)
            z = jnp.log(jnp.abs(x))
            # print('X aft log', z)
            z_kernel = self.param(f'z_kernel_{i}', nn.initializers.lecun_normal(), (x.shape[-1], proj_dim))
            z_bias = self.param(f'z_bias_{i}', nn.initializers.zeros_init(), (proj_dim,))
            # print('Z_SHAP', z.shape)
            # print('Z_KER_SHAP', z_kernel.shape)
            # print('Z_BIAS_SHAP', z_bias.shape)

            z = z @ z_kernel + z_bias
            # print('Z aft DENSE', z)
            sign = jnp.cos(jnp.pi * jnp.sum(z_kernel, axis=0))
            z = jnp.exp(z) * sign
            # print('OUT Z', z)

            x = jnp.concatenate((x_lin, z), axis=-1)
    
        out = nn.Dense(1)(x).flatten()
        # print('OUT', out)
        return out


if __name__ == '__main__':
    model = PolyConfig(10).to_model()
    print(model.tabulate(jax.random.key(10), jnp.ones((32, 2)).astype(jnp.int32)))
    # params = model.init(jax.random.key(0), jnp.ones((32, 2)).astype(jnp.int32))
    # print(jax.tree_map(lambda x: x.shape, params))