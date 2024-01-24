"""
Simple MLP model
"""
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
            x = nn.relu(x)
    
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
    use_quadratic_activation: bool = True

    def to_model(self):
        return MLP(self)


class RF(nn.Module):

    config: MlpConfig

    def setup(self):
        self.w_rf = np.random.randn(self.config.n_in, self.config.n_hidden)
        # TODO: finish implement with w_rf below

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)

        for _ in range(self.config.n_layers):
            x = nn.Dense(self.config.n_hidden)(x)
            x = nn.relu(x)
    
        out = nn.Dense(self.config.n_out)(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out