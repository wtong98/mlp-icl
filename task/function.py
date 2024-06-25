"""
Tasks for approximating different functions

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np


class PointTask:
    def __init__(self, points, batch_size=128) -> None:
        xs, ys = zip(*points)
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.batch_size = batch_size
    
    def __next__(self):
        if self.batch_size is None:
            idxs = np.arange(len(self.xs))
        else:
            idxs = np.random.choice(len(self.xs), size=self.batch_size, replace=True)
        return self.xs[idxs], self.ys[idxs]

    def __iter__(self):
        return self


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


class MultiplicationTask:
    def __init__(self, domain, batch_size=128) -> None:
        self.lower, self.upper = domain
        self.batch_size = batch_size
    
    def __next__(self):
        xs = np.random.uniform(self.lower, self.upper, size=(self.batch_size, 2))
        ys = xs[:,0] * xs[:,1]
        return xs, ys

    def __iter__(self):
        return self


class DotProductTask:
    def __init__(self, domain, n_args=2, n_dims=5, batch_size=128):
        self.lower, self.upper = domain
        self.n_args = n_args
        self.n_dims = n_dims
        self.batch_size = batch_size
    
    def __next__(self):
        xs = np.random.uniform(self.lower, self.upper, size=(self.batch_size, self.n_args, self.n_dims))
        # ys = np.diag(xs[:,0] @ xs[:,1].T)
        ys = np.sum(np.prod(xs, axis=1), axis=-1)
        return xs, ys
    
    def __iter__(self):
        return self


class AttentionTask:
    def __init__(self, domain, n_dims=5, batch_size=128):
        self.lower, self.upper = domain
        self.n_dims = n_dims
        self.batch_size = batch_size
    
    def __next__(self):
        xs = np.random.uniform(self.lower, self.upper, size=(self.batch_size, 2, self.n_dims))
        ys = np.diag(xs[:,0] @ xs[:,1].T)
        return xs, ys
    
    def __iter__(self):
        return self


def batch_choice(a, n_elem, batch_size, rng=None):
    assert n_elem <= len(a), f'require n_elem <= len(a), got n_elem={n_elem} and len(a)={len(a)}'

    if rng is None:
        rng = np.random.default_rng(None)

    idxs = np.tile(a, (batch_size, 1))
    idxs = rng.permuted(idxs, axis=1)
    idxs = idxs[:,:n_elem]
    return idxs


class SameDifferent:
    def __init__(self, n_symbols=None, task='hard',
                 n_dims=2, thresh=0, radius=1,    # soft/hard params
                 n_seen=None, sample_seen=True,   # token params
                 seed=None, reset_rng_for_data=True, batch_size=128) -> None:

        if task == 'token':
            assert n_symbols is not None and n_symbols >= 4, 'if task=token, n_symbols should be >= 4'
            
            if n_seen is None:
                n_seen = n_symbols // 2

        self.n_symbols = n_symbols
        self.task = task
        self.n_dims = n_dims
        self.thresh = thresh
        self.radius = radius
        self.n_seen = n_seen
        self.sample_seen = sample_seen
        self.seed = seed
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
        if self.n_symbols is not None:
            self.symbols = self.rng.standard_normal((self.n_symbols, self.n_dims)) / np.sqrt(self.n_dims)

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        if self.task == 'soft':
            return self._sample_soft()
        elif self.task == 'hard':
            return self._sample_hard()
        elif self.task == 'token':
            return self._sample_token()
        else:
            raise ValueError(f'unrecognized task type: {self.task}')

    def _sample_soft(self):
        xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims)) / np.sqrt(self.n_dims)
        # norms = np.linalg.norm(xs, axis=-1, keepdims=True)
        # xs = xs / norms * self.radius

        x0, x1 = xs[:,0], xs[:,1]
        ys = (np.einsum('bi,bi->b', x0, x1) > self.thresh).astype('float')
        return xs, ys.flatten()
    
    def _sample_hard(self):
        if self.n_symbols is None:
            xs = self.rng.standard_normal((self.batch_size, 2, self.n_dims)) / np.sqrt(self.n_dims)
        else:
            sym_idxs = batch_choice(np.arange(self.n_symbols), 2, batch_size=self.batch_size, rng=self.rng)
            xs = self.symbols[sym_idxs]

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys
    
    def _sample_token(self):
        if self.sample_seen:
            xs = batch_choice(np.arange(0, self.n_seen), 2, batch_size=self.batch_size, rng=self.rng)
        else:
            xs = batch_choice(np.arange(self.n_seen, self.n_symbols), 2, batch_size=self.batch_size, rng=self.rng)

        ys = self.rng.binomial(n=1, p=0.5, size=(self.batch_size,))
        if np.sum(ys) > 0:
            idxs = ys.astype(bool)
            xs[idxs,1] = xs[idxs,0]

        return xs, ys

    def __iter__(self):
        return self


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    task = SameDifferent(batch_size=2048, seed=4, n_symbols=16, n_seen=2, task='token')
    xs, ys = next(task)
    print(xs)
    print(ys)
    # print(np.mean(ys))
    ys_true = (xs[:,0] == xs[:,1]).astype(int)
    # ys_true = (xs[:,0] == xs[:,1]).mean(axis=1).astype(int)
    print(np.mean(ys_true == ys))
    

    # x = xs[0]
    # y = ys[0]

    # print(np.sum(x[0] * x[1]))
    # print(y)

    # task = PowerTask(apply_random_token_proj=False, tokenize=1)
    # xs, ys = next(task)
    # print(xs.shape)

    # task = ClassificationTask(n_dims=2, n_classes=5, tokenize=2)
    # xs, ys = next(task)
    # print(xs.shape)

    # xs = xs.reshape(128, -1)
    
    # plt.scatter(xs[:,0], xs[:,1], c=ys)
    # plt.scatter(task.centers[:,0], task.centers[:,1], color='red')
    
    # print(task.centers)


# %%
