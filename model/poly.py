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

    def _fwd_product_only_positive(self, x):
        z_kernel = self.param(f'z_kernel', nn.initializers.lecun_normal(), (x.shape[-1], 1))
        z = jnp.log(jnp.abs(x))
        z = z @ z_kernel
        z = jnp.exp(z)
        return z.flatten()
    
    def _fwd_product_only_cos_negative(self, x):
        z_kernel = self.param(f'z_kernel', nn.initializers.lecun_normal(), (x.shape[-1], 1))
        z = jnp.log(jnp.abs(x))
        z = z @ z_kernel
        neg_powers = z_kernel.T * (x < 0).astype(int)
        sign = jnp.cos(jnp.pi * jnp.sum(neg_powers, axis=1))
        z = jnp.exp(z) * jnp.expand_dims(sign, 1)  # NOTE: cosine generalization seems to be ineffectual
        return z.flatten()
    
    def _fwd_product_mlp_parallel_cos(self, x):
        if self.n_hidden % 2 != 0:
            raise ValueError(f'n_hidden should be even, got n_hidden={self.n_hidden}')

        proj_dim = self.config.n_hidden // 2

        for i in range(self.config.n_layers):
            x_lin = nn.Dense(proj_dim)(x)
            z = jnp.log(jnp.abs(x))

            z_kernel = self.param(f'z_kernel_{i}', nn.initializers.lecun_normal(), (x.shape[-1], proj_dim))
            z_bias = self.param(f'z_bias_{i}', nn.initializers.zeros_init(), (proj_dim,))

            z = z @ z_kernel + z_bias
            neg_powers = z_kernel.T * (x < 0).astype(int)
            sign = jnp.cos(jnp.pi * jnp.sum(neg_powers, axis=1))
            z = jnp.exp(z) * jnp.expand_dims(sign, 1)
            x = jnp.concatenate((x_lin, z), axis=-1)
    
        out = nn.Dense(1)(x).flatten()
        return out
    
    def _fwd_product_mlp_seq_positive(self, x):
        # NOTE: fails to learn anything meaningful, particularly on negative domain
        hid = self.config.n_hidden

        x = nn.Dense(hid)(x)
        x = nn.relu(x) + 1e-8

        x = jnp.log(x)
        x = nn.Dense(hid)(x)
        x = jnp.exp(x)

        x = nn.Dense(hid)(x)
        x = nn.relu(x)

        x = nn.Dense(1)(x).flatten()
        return x

    def m_block(self, x, n_features=None):
        if n_features is None:
            n_features = self.config.n_hidden

        z = jnp.log(jnp.abs(x))
        z = nn.Dense(n_features, name='DenseMult')(z)
        z = jnp.exp(z)

        beta = 10 # NOTE: sharpen tanh, can make trainable
        s = nn.tanh(beta * x)

        # NOTE: new product feature style
        s_prod = jnp.einsum('ij,ik->ijk', s, s).reshape(s.shape[0], -1)
        s = jnp.concatenate((s, s_prod), axis=-1)
        s = nn.Dense(n_features, name='SignOut')(s)
        s = nn.tanh(s)

        # NOTE: old 2-layer NN style
        # s = nn.Dense(self.config.n_hidden, name='SignHid')(s)  # TODO: make separate param
        # s = nn.gelu(s)
        # s = nn.Dense(n_features, name='SignOut')(s)
        # s = nn.tanh(0.5 * s) # NOTE: can also add sigmoid term for zeroing

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