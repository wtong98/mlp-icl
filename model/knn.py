"""
K-Nearest Neighbors model
"""

# <codecell>
from dataclasses import dataclass
import numpy as np

import jax
import jax.numpy as jnp

# @jax.jit
def compute_dists(point, data):
    '''
    point: B x D
    data:  N x D
    '''
    diff = data - jnp.expand_dims(point, axis=1)
    dot = jnp.einsum('bnd,bmd->bnm', diff, diff)

    dists = jnp.sqrt(jnp.diagonal(dot, axis1=1, axis2=2))
    return dists


@dataclass
class KnnConfig:
    """Global hyperparamters"""
    beta: float = 1
    n_classes: int | None = None
    xs: np.ndarray = None
    ys: np.ndarray = None

    def to_model(self):
        return Knn(self)


class Knn:
    def __init__(self, config) -> None:
        self.config = config
    
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)

        dists = compute_dists(x, self.config.xs)
        weights = jnp.exp(-self.config.beta * dists)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        if self.config.n_classes is None:
            return jnp.sum(weights * self.config.ys, axis=1)
        else:
            props = jnp.stack([jnp.sum(weights[:,y==self.config.ys], axis=1) for y in np.arange(self.config.n_classes)], axis=1)
            return props
        
    
if __name__ == '__main__':
    # point = np.array([[0, 0], [0.01, 0.01]])
    # data = np.array([[0, 0], [-1, -1], [1, 1,]])

    # compute_dists(point, data)

    point = np.array([[0, 0], [0.01, 1]])
    data = np.array([[0, 0], [-1, -1], [1, 1,]])
    labs = np.array([0, 1, 2])

    config = KnnConfig(n_classes=3, beta=4, xs=data, ys=labs)
    model = config.to_model()
    model(point)

