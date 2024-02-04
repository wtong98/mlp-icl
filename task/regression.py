"""
In-context regression tasks

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>

import numpy as np
from scipy.stats import special_ortho_group

class LinearRegression:
    def __init__(self, n_points=6, n_dims=2, x_range=(-5, 5), w_scale=1, batch_size=128, seed=None) -> None:
        self.n_points = n_points
        self.n_dims = n_dims
        self.x_range = x_range
        self.w_scale = w_scale
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
    
    def __next__(self):
        xs = self.rng.uniform(*self.x_range, size=(self.batch_size, self.n_points, self.n_dims))
        ws = self.rng.normal(loc=0, scale=self.w_scale, size=(self.batch_size, self.n_dims, 1))
        ys = xs @ ws
        zs = np.zeros((self.batch_size, self.n_points, self.n_dims - 1))
        ys_pad = np.concatenate((ys, zs), axis=-1)

        interl_xs = np.empty((self.batch_size, self.n_points * 2 - 1, self.n_dims))
        interl_xs[:, 0::2] = xs
        interl_xs[:, 1::2] = ys_pad[:,:-1]

        return interl_xs, ys[:,-1].squeeze()


    def __iter__(self):
        return self


class FiniteLinearRegression:
    """Based on the construction described in Raventos et al. 2023"""

    def __init__(self, n_ws=128, n_points=16, n_dims=8, noise_scale=0.5, batch_size=128, stack_y=True, enforce_orth_x=False, seed=None, reset_rng_for_data=True) -> None:
        self.n_points = n_points
        self.n_dims = n_dims
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_ws = n_ws
        self.stack_y = stack_y
        self.enforce_orth_x = enforce_orth_x

        self.ws = None
        if n_ws is not None:
            self.ws = self.rng.standard_normal(size=(n_ws, n_dims))
        
        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        if self.enforce_orth_x:
            xs = np.expand_dims(np.identity(self.n_dims), axis=0)
            # xs = np.repeat(xs, self.batch_size, axis=0)

            Qs = special_ortho_group.rvs(self.n_dims, size=self.batch_size)
            xs = Qs @ xs

            rand_coef = self.rng.standard_normal(size=(self.batch_size, self.n_dims, 1))
            # xs_q = np.ones((self.batch_size, 1, self.n_dims))
            xs_q = np.sum(rand_coef * xs, axis=1, keepdims=True)
            xs = np.concatenate((xs, xs_q), axis=1)
        else:
            xs = self.rng.standard_normal(size=(self.batch_size, self.n_points, self.n_dims))

        if self.ws is None:
            ws = self.rng.standard_normal(size=(self.batch_size, self.n_dims))
        else:
            ws_idxs = self.rng.choice(len(self.ws), size=(self.batch_size), replace=True)
            ws = self.ws[ws_idxs]
        
        ws = np.expand_dims(ws, axis=-1)
        ys = xs @ ws + self.rng.normal(scale=self.noise_scale, size=(self.batch_size, xs.shape[1], 1))

        if self.stack_y:
            out = np.concatenate((xs, ys), axis=-1)
            ys_true = ys[:,-1].squeeze()
            out[:, -1, -1] = 0
            return out, ys_true
        else:
            zs = np.zeros((self.batch_size, self.n_points, self.n_dims - 1))
            ys_pad = np.concatenate((ys, zs), axis=-1)

            interl_xs = np.empty((self.batch_size, self.n_points * 2 - 1, self.n_dims))
            interl_xs[:, 0::2] = xs
            interl_xs[:, 1::2] = ys_pad[:,:-1]
            return interl_xs, ys[:,-1].squeeze()


    def __iter__(self):
        return self

def t(xs):
    return np.swapaxes(xs, -2, -1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # task = LinearRegression(batch_size=5, n_dims=1)
    # xs, ys = next(task)

    # plt.scatter(xs[0][0:-1:2], xs[0][1::2])
    # plt.scatter([xs[0][-1]], ys[0])

    # task = LinearRegression(batch_size=5, n_dims=2, n_points=500, seed=1)
    # xs, ys = next(task)

    task = FiniteLinearRegression(n_ws=None, batch_size=5, n_dims=2, n_points=500, seed=1, enforce_orth_x=True, noise_scale=0)
    xs, ys = next(task)
    print('XS', xs.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # ax.scatter(xs[0][0:-1:2, 0], xs[0][0:-1:2, 1], xs[0][1::2, 0], alpha=0.3)
    # ax.scatter(xs[0][-1,0], xs[0][-1, 1], ys[0])

    ax.scatter(xs[0][:-1,0], xs[0][:-1, 1], xs[0][:-1, 2], alpha=0.3)
    ax.scatter(xs[0][-1,0], xs[0][-1, 1], ys[0])

    # print(xs @ t(xs))
    x = xs[0,:2,:2]
    y = xs[0,:2,[-1]]
    x_q = xs[0,[-1],:-1]
    print(y @ x @ x_q.T)
    print(ys[0])

