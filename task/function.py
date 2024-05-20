"""
Simple tasks
"""

import numpy as np

class PowerTask:
    def __init__(self, n_dims=16, eta=0.05, power=1, seed=None, reset_rng_for_data=True, tokenize=False, apply_random_token_proj=False, batch_size=128) -> None:
        self.n_dims = n_dims
        self.eta = eta
        self.power = power
        self.seed = seed
        self.batch_size = batch_size
        self.tokenize = tokenize
        self.apply_random_token_proj = apply_random_token_proj

        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.standard_normal(size=(self.n_dims, 1)) / np.sqrt(self.n_dims)
        self.rand_proj = self.rng.standard_normal(size=(1, 128)) / np.sqrt(128)

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        xs = self.rng.standard_normal(size=(self.batch_size, self.n_dims))
        ys = (xs @ self.weights)**self.power + self.rng.standard_normal(size=(self.batch_size, 1)) * np.sqrt(self.eta)

        if self.tokenize:
            try:
                xs = np.reshape(xs, (self.batch_size, -1, self.tokenize))
                if self.apply_random_token_proj:
                    xs = xs @ self.rand_proj

            except TypeError:  # self.tokenize is not an integer
                xs = np.expand_dims(xs, axis=-1)

        return xs, ys.flatten()

    def __iter__(self):
        return self


class ClassificationTask:
    def __init__(self, n_classes=2, n_dims=16, seed=None, reset_rng_for_data=True, tokenize=False, batch_size=128) -> None:
        self.n_classes = n_classes
        self.n_dims = n_dims
        self.seed = seed
        self.reset_rng_for_data = reset_rng_for_data
        self.tokenize = tokenize
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.centers = self.rng.standard_normal(size=(self.n_classes, self.n_dims)) / np.sqrt(self.n_dims)

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        xs = self.rng.standard_normal(size=(self.batch_size, self.n_dims))
        dists = np.linalg.norm(np.expand_dims(xs, axis=1) - self.centers, axis=-1)  # n_batches x n_classes
        ys = np.argmin(dists, axis=-1)

        if self.tokenize:
            try:
                xs = np.reshape(xs, (self.batch_size, -1, self.tokenize))
            except TypeError:  # self.tokenize is not an integer
                xs = np.expand_dims(xs, axis=-1)

        return xs, ys

    def __iter__(self):
        return self


class SameDifferent:
    def __init__(self, n_dims=2, soft=True, seed=None, reset_rng_for_data=True, batch_size=128) -> None:
        self.n_dims = n_dims
        self.soft = soft
        self.seed = seed
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        if self.soft:
            return self._sample_soft()
        if not self.soft:
            return self._sample_hard()

    def _sample_soft(self):
        xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims))
        norms = np.linalg.norm(xs, axis=-1, keepdims=True)
        xs = xs / norms

        x0, x1 = xs[:,0], xs[:,1]
        ys = (np.einsum('bi,bi->b', x0, x1) > 0).astype('float')
        return xs, ys.flatten()
    
    def _sample_hard(self):
        xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims))
        norms = np.linalg.norm(xs, axis=-1, keepdims=True)
        xs = xs / norms

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys

    def __iter__(self):
        return self


class SameDifferentToken:
    def __init__(self, n_vocab=16, n_seen=8, sample_seen=True, seed=None, reset_rng_for_data=True, batch_size=128) -> None:
        assert n_seen <= n_vocab

        self.n_seen = n_seen
        self.n_vocab = n_vocab
        self.sample_seen = sample_seen

        self.seed = seed
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        if self.sample_seen:
            xs = self.rng.integers(low=0, high=self.n_seen, size=(self.batch_size, 2))
        else:
            xs = self.rng.integers(low=self.n_seen, high=self.n_vocab, size=(self.batch_size, 2))

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys

    def __iter__(self):
        return self
    