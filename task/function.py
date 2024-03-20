"""
Tasks for approximating different functions

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np


class PointTask:
    def __init__(self, points, batch_size=128) -> None:
        self.points = np.array(points)
        self.batch_size = batch_size
    
    def __next__(self):
        idxs = np.random.choice(len(self.points), size=self.batch_size, replace=True)
        xs, ys = zip(*self.points[idxs])
        return np.array(xs), np.array(ys)

    def __iter__(self):
        return self


class PowerTask:
    def __init__(self, n_dims=16, eta=0.05, power=1, seed=None, reset_rng_for_data=True, tokenize=False, batch_size=128) -> None:
        self.n_dims = n_dims
        self.eta = eta
        self.power = power
        self.seed = seed
        self.batch_size = batch_size
        self.tokenize = tokenize

        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.standard_normal(size=(self.n_dims, 1)) / np.sqrt(self.n_dims)

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        xs = self.rng.standard_normal(size=(self.batch_size, self.n_dims))
        ys = (xs @ self.weights)**self.power + self.rng.standard_normal(size=(self.batch_size, 1)) * np.sqrt(self.eta)

        if self.tokenize:
            try:
                xs = np.reshape(xs, (self.batch_size, -1, self.tokenize))
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
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    task = ClassificationTask(n_dims=2, n_classes=5, tokenize=2)
    xs, ys = next(task)
    print(xs.shape)

    xs = xs.reshape(128, -1)
    
    plt.scatter(xs[:,0], xs[:,1], c=ys)
    plt.scatter(task.centers[:,0], task.centers[:,1], color='red')
    
    print(task.centers)


# %%
