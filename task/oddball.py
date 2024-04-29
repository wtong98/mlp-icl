"""
Oddball task variants, inspired from Sable-Meyer's geometric Oddball perceptual
tasks

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np


class FreeOddballTask:
    def __init__(self, n_choices=6, 
                 discrim_dist=5, box_radius=10, n_dims=2, 
                 batch_size=128, 
                 data_size=None, n_retry_if_missing_labels=0, 
                 one_hot=False, 
                 seed=None, reset_rng_for_data=False) -> None:

        self.n_choices = n_choices
        self.discrim_dist = discrim_dist
        self.box_radius = box_radius
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.data_size = data_size
        self.one_hot = one_hot
        self.rng = np.random.default_rng(seed)

        if data_size is not None:
            self.data = self._samp_batch(data_size)

            for i in range(n_retry_if_missing_labels):
                labels = self.data[1]
                if len(np.unique(labels)) == n_choices:
                    break

                print('warn: retry number', i)
                self.data = self._samp_batch(data_size)
        
        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)

    def _samp_batch(self, size):
        centers = self.rng.uniform(-self.box_radius, self.box_radius, size=(size, 1, self.n_dims))
        points = self.rng.standard_normal(size=(size, self.n_choices, self.n_dims))

        angles = self.rng.uniform(0, 2 * np.pi, size)
        oddballs = self.discrim_dist * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        oddball_idxs = self.rng.choice(self.n_choices, size=size, replace=True)

        points[np.arange(size), oddball_idxs] += oddballs
        xs = centers + points
        
        if self.one_hot:
            z = np.zeros((size, self.n_choices))
            z[np.arange(size), oddball_idxs] = 1
            oddball_idxs = z

        return xs, oddball_idxs
    
    def __next__(self):
        if self.data_size is None:
            return self._samp_batch(self.batch_size)
        else:
            idxs = self.rng.choice(self.data_size, size=self.batch_size, replace=True)
            return self.data[0][idxs], self.data[1][idxs]

    def __iter__(self):
        return self


class LineOddballTask:
    def __init__(self, n_choices=6, linear_dist=1, perp_dist=1, batch_size=128, with_dot_product_feats=False) -> None:
        self.n_choices = n_choices
        self.linear_dist = linear_dist
        self.perp_dist = perp_dist
        self.batch_size = batch_size
        self.with_dot_product_feats = with_dot_product_feats

    def __next__(self):
        dirs = np.random.uniform(0, 2 * np.pi, size=self.batch_size)
        perp_dirs = dirs + np.random.choice([1, -1], size=self.batch_size) * np.pi / 2
        radii = np.random.normal(0, self.linear_dist, size=(self.batch_size, self.n_choices))

        radii = np.expand_dims(radii, axis=-1)
        angles = np.expand_dims(np.stack([np.cos(dirs), np.sin(dirs)], axis=-1), axis=1)
        points = radii * angles

        oddballs = self.perp_dist * np.stack([np.cos(perp_dirs), np.sin(perp_dirs)], axis=-1)
        oddball_idxs = np.random.choice(self.n_choices, size=self.batch_size, replace=True)

        points[np.arange(self.batch_size), oddball_idxs] = oddballs

        if self.with_dot_product_feats:
            points = points - np.mean(points, axis=1, keepdims=True)
            points = points / np.linalg.norm(points, axis=-1, keepdims=True)
            points = points @ np.transpose(points, axes=(0, 2, 1))

        return points, oddball_idxs

    def __iter__(self):
        return self


class FixedOddballTask:
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # task = FreeOddballTask(reset_rng_for_data=True, n_choices=20)
    task = LineOddballTask(n_choices=20, perp_dist=1, linear_dist=1)

    fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.5))

    for ax, (xs, ys) in zip([axs], task):
        # c = np.zeros(6)
        # c[ys[0]] = 0

        ax.scatter(xs[0,:,0], xs[0,:,1], c=np.zeros(20))
        ax.axis('equal')
        ax.set_axis_off()
        # plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('../experiment/fig/line_oddball_example.svg')
    # plt.savefig('../experiment/fig/line_oddball_examples.png')
