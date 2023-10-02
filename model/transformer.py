"""
Adapted from: https://github.com/google/flax/blob/main/examples/lm1b

License notice:
Copyright 2023 The Flax Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# <codecell>
import functools

from flax import linen as nn, struct

import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class TransformerConfig:
    vocab_size: int | None = None
    n_layers: int = 2
    n_emb: int = 128
    max_len: int = 2
    pos_emb: bool = False
    return_final_logits_only: bool = True

    def to_model(self):
        return Transformer(self)


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0,
                    squeeze=False):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
            max_len: maximum possible length for the input.
            min_scale: float: minimum frequency-scale in sine grating.
            max_scale: float: maximum frequency-scale in sine grating.

    Returns:
            output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, :d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)

        if not squeeze:
            pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]

        return jnp.array(pe)

    return init


class SingleHeadSelfAttention(nn.Module):
    """Single head self attention, with some custom sauce.
    
    Args:
        config: TransformerConfig dataclass with hyperparameters
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, mask=None, idxs=None, use_bias=False):
        dense = functools.partial(
            nn.Dense,
            features=self.config.n_emb,
            use_bias=use_bias)
        
        self.sow('intermediates', 'inputs', inputs)
        query = dense(name='query')(inputs)
        key = dense(name='key')(inputs)
        value = dense(name='value')(inputs)
        depth = query.shape[-1]

        attn_weights = jnp.einsum('...qd,...kd->...qk', query, key)
        attn_weights /= jnp.sqrt(depth)
        self.sow('intermediates', 'raw_att', attn_weights)

        if mask is not None:
            attn_weights = jnp.where(mask.squeeze(), attn_weights, np.iinfo(np.int32).min)

        attn_weights = jax.nn.softmax(attn_weights)
        self.sow('intermediates', 'attention_weights', attn_weights)

        attn_out = attn_weights @ value
        return attn_out


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Args:
            inputs: input data.

        Returns:
            output: `(bs, timesteps, in_dim)`
        """
        config = self.config
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                 ' but it is: %d' % inputs.ndim)
        length = inputs.shape[1]
        pos_emb_shape = (1, config.max_len, inputs.shape[-1])
        pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
                                                                pos_emb_shape,
                                                                None)

        # if config.posemb_init is None:
        #     # Use a fixed (non-learned) sinusoidal position embedding.
        #     pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
        #                                                             pos_emb_shape,
        #                                                             None)
        # else:
        #     pos_embedding = self.param('pos_embedding', config.posemb_init, pos_emb_shape)
        
        pe = pos_embedding[:, :length, :]
        return inputs + pe


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self,
                inputs,
                decoder_mask=None,
                idxs=None):

        assert inputs.ndim == 3
        x = SingleHeadSelfAttention(self.config)(inputs, decoder_mask, idxs=idxs)
        x = x + inputs

        return x


class Transformer(nn.Module):

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):

        config = self.config
        assert inputs.ndim == 2  # (batch, len)

        # Target Embedding
        y = nn.Embed(
                num_embeddings=config.vocab_size,
                features=config.n_emb)(inputs)

        if config.pos_emb:
            y = AddPositionEmbs(config=config)(y)
        
        # decoder_mask = nn.make_attention_mask(inputs > 0, inputs > 0)
        # decoder_mask = nn.combine_masks(
        #     decoder_mask,
        #     nn.make_causal_mask(inputs))
        decoder_mask = nn.make_causal_mask(inputs)
        
        for _ in range(config.n_layers):
            y = TransformerBlock(
                config=config)(
                        y,
                        decoder_mask=decoder_mask)

        logits = nn.Dense(1)(y)
        if config.return_final_logits_only:
            logits = logits[:,-1,:].flatten()
        return logits

