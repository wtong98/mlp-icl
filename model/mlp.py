"""
Simple MLP model
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct

def parse_act_fn(fn: str):
    if fn == 'relu':
        return jax.nn.relu
    elif fn == 'linear':
        return lambda x: x
    elif fn == 'gelu':
        return jax.nn.gelu
    elif fn =='quadratic':
        return lambda x: x**2
    else:
        raise ValueError(f'function not recognized: {fn}')


@struct.dataclass
class MlpConfig:
    """Global hyperparamters"""
    vocab_size: int | None = None
    n_layers: int = 2
    n_emb: int = 64
    n_hidden: int = 128
    n_out: int = 1
    act_fn: str = 'relu'
    layer_norm: bool = False

    def to_model(self):
        return MLP(self)


class MLP(nn.Module):

    config: MlpConfig

    @nn.compact
    def __call__(self, x):
        act_fn = parse_act_fn(self.config.act_fn)

        if self.config.vocab_size is not None:
            x = nn.Embed(
                num_embeddings=self.config.vocab_size,
                features=self.config.n_emb)(x)
        
        x = x.reshape(x.shape[0], -1)

        for _ in range(self.config.n_layers):
            x = nn.Dense(self.config.n_hidden)(x)

            if self.config.layer_norm:
                x = nn.LayerNorm()(x)

            x = act_fn(x)
    
        out = nn.Dense(self.config.n_out)(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out


@struct.dataclass
class SpatialMlpConfig:
    """Global hyperparamters"""
    n_layers: int = 2
    n_hidden: int = 128
    n_channels: int = 16
    n_out: int = 1
    act_fn: str = 'relu'
    last_token_only: bool = True
    layer_norm: bool = False

    def to_model(self):
        return SpatialMLP(self)


class SpatialMLP(nn.Module):

    config: SpatialMlpConfig

    @nn.compact
    def __call__(self, x):
        act_fn = parse_act_fn(self.config.act_fn)
        assert len(x.shape) == 3

        for _ in range(self.config.n_layers):
            x = nn.Dense(self.config.n_hidden)(x)
            x = jnp.transpose(x, (0, 2, 1))

            x = nn.Dense(self.config.n_channels)(x)
            x = jnp.transpose(x, (0, 2, 1))

            if self.config.layer_norm:
                x = nn.LayerNorm()(x)

            x = act_fn(x)
    
        if self.config.last_token_only:
            x = x[:,-1,:]
        else:
            x = x.reshape(x.shape[0], -1)

        out = nn.Dense(self.config.n_out)(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out


@struct.dataclass
class DotMlpConfig:
    """Global hyperparamters"""
    n_hidden: int = 128
    n_out: int = 1
    use_initial_proj: bool = True
    last_token_only: bool = False

    def to_model(self):
        return DotMLP(self)


class DotMLP(nn.Module):
    
    config: DotMlpConfig

    @nn.compact
    def __call__(self, x):
        if self.config.use_initial_proj:
            x = nn.Dense(self.config.n_hidden)(x)  # B x L x H

        x = jnp.einsum('...ih,...jh->...ij', x, x) # B x L x L

        if self.config.last_token_only:
            x = x[:,-1,:]
        else:
            x = x.reshape(x.shape[0], -1)

        # x = nn.Dense(self.config.n_hidden)(x)
        # x = nn.relu(x)
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
        # x = x @ self.w_rf2

        if self.config.use_quadratic_activation:
            x = x**2

        out = self.readout(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out