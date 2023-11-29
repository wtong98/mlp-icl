"""
Oddball task variants, inspired from Sable-Meyer's geometric Oddball perceptual
tasks

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np


class FreeOddballTask:
    def __init__(self, n_choices=6, discrim_dist=5, box_radius=10, n_dims=2, batch_size=128, data_size=None) -> None:
        self.n_choices = n_choices
        self.discrim_dist = discrim_dist
        self.box_radius = box_radius
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.data_size = data_size

        if data_size is not None:
            self.data = self._samp_batch(data_size)

    
    def _samp_batch(self, size):
        centers = np.random.uniform(-self.box_radius, self.box_radius, size=(size, 1, self.n_dims))
        points = np.random.randn(size, self.n_choices, self.n_dims)

        angles = np.random.uniform(0, 2 * np.pi, size)
        oddballs = self.discrim_dist * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        oddball_idxs = np.random.choice(self.n_choices, size=size, replace=True)

        points[np.arange(size), oddball_idxs] += oddballs
        xs = centers + points

        return xs, oddball_idxs
    
    def __next__(self):
        if self.data_size is None:
            return self._samp_batch(self.batch_size)
        else:
            idxs = np.random.choice(self.data_size, size=self.batch_size, replace=True)
            return self.data[0][idxs], self.data[1][idxs]
            

    def __iter__(self):
        return self

class FixedOddballTask:
    pass


if __name__ == '__main__':
    task = FreeOddballTask(data_size=1024)
    xs, ys = next(task)

    import matplotlib.pyplot as plt
    plt.scatter(xs[0,:,0], xs[0,:,1], c=np.arange(6))
    plt.gca().axis('equal')
    plt.colorbar()
    print(ys)
