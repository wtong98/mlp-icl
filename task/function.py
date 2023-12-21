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


class TripleScalarProductTask:
    pass


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
    points = [(0, 0), (1, 2)]
    task = PointTask(points, batch_size=5)
    xs, y = next(task)
    print(xs)
    print(y)
    # task = DotProductTask(domain=(-5, 5))
    # xs, ys = next(iter(task))

    # print('XS', xs.shape)
    # print('YS', ys.shape)
    # print(ys[1])
    # print(xs[1,[0]] @ xs[1, [1]].T)


# %%
