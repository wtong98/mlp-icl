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
    n_out: int = 1
    start_with_dense: bool = True
    use_mult_signage: bool = False
    learnable_signage_temp: bool = True
    disable_signage: bool = False
    sharpen_signange_inputs: bool = False

    def to_model(self):
        return PolyNet(self)


class MBlock(nn.Module):
    config: PolyConfig
    n_features: int | None = None

    @nn.compact
    def __call__(self, x):
        if self.n_features is None:
            self.n_features = self.config.n_hidden

        z = jnp.log(jnp.abs(x) + 1e-32)
        z = nn.Dense(self.n_features, 
                     kernel_init=nn.initializers.variance_scaling(scale=0.1, mode='fan_out', distribution='truncated_normal'),
                     name='DenseMultiply')(z)
        z = jnp.exp(z)

        if self.config.disable_signage:
            return z

        if self.config.sharpen_signange_inputs:
            beta = 10 # NOTE: sharpen tanh, can make trainable
            s = nn.tanh(beta * x)
        else:
            s = x

        # NOTE: need n-power features to capture n-power signs
        if self.config.use_mult_signage:
            s_prod = jnp.einsum('ij,ik->ijk', s, s).reshape(s.shape[0], -1)
            s_prod = jnp.einsum('ij,ik->ijk', s_prod, s).reshape(s.shape[0], -1)
            s = jnp.concatenate((s, s_prod), axis=-1)
        else:
            s = nn.Dense(self.config.n_hidden)(s)
            s = nn.gelu(s)

        s = nn.Dense(self.n_features)(s)

        if self.config.learnable_signage_temp:
            out_temp = self.param('OutputTemperature', nn.initializers.constant(0), (1,))
        else:
            out_temp = 1

        s = nn.tanh(s * out_temp)

        out = s * z
        return out


class PolyNet(nn.Module):

    config: PolyConfig

    def poly_block(self, x, n_features=None):
        if n_features is None:
            n_features = self.config.n_hidden

        x = MBlock(self.config, n_features=self.config.n_hidden)(x)
        x = nn.Dense(n_features)(x)
        return x

    def _fwd_product_sep_sign_full(self, x):
        if self.config.start_with_dense:
            x = nn.Dense(self.config.n_hidden)(x)

        for _ in range(self.config.n_layers - 1):
            x = self.poly_block(x, n_features=self.config.n_hidden)

        out = self.poly_block(x, n_features=self.config.n_out)

        # NOTE: somehow appending a dense layer is disastrous for performance
        # out = self.poly_block(x, n_features=self.config.n_hidden)
        # out = nn.Dense(self.config.n_out)(x)
        if self.config.n_out == 1:
            out = out.flatten()
        
        return out


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