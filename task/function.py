"""
Tasks for approximating different functions

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np


class MultiplicationTask:
    def __init__(self, domain, batch_size=32) -> None:
        self.lower, self.upper = domain
        self.batch_size = batch_size
    
    def __next__(self):
        xs = np.random.uniform(self.lower, self.upper, size=(self.batch_size, 2))
        ys = xs[:,0] * xs[:,1]
        return xs, ys

    def __iter__(self):
        return self


class DotProductTask:
    def __init__(self, domain, n_dims=5, batch_size=32):
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
    task = DotProductTask(domain=(-5, 5))
    xs, ys = next(iter(task))

    print('XS', xs.shape)
    print('YS', ys.shape)
    print(ys[1])
    print(xs[1,[0]] @ xs[1, [1]].T)

