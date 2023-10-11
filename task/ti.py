"""
A simple transitive inference task

Paper implementation: https://github.com/sflippl/relational-generalization-in-ti/tree/main

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np


def enumerate_pairs(n_items, dist):
    if dist > n_items - 1:
        raise ValueError(f'dist={dist} is too large for n_items={n_items}')
    
    idxs = np.arange(n_items)
    partner = idxs + dist
    valid_idxs = partner < n_items

    return np.stack((idxs[valid_idxs], partner[valid_idxs])).astype(np.int32)


class TiTask:
    def __init__(self, n_items=5, n_dims=100, dist=1, dist_p=None, batch_size=32) -> None:
        self.rep = np.random.randn(n_items, n_dims)

        try:
            self.dist = list(dist)
        except TypeError:
            self.dist = [dist]
        
        if dist_p is None:
            dist_p = np.ones(len(self.dist))
        self.dist_p = dist_p / np.sum(dist_p)

        self.batch_size = batch_size
        self.all_pairs = {d: enumerate_pairs(n_items, d) for d in self.dist}
    
    def __next__(self):
        dists = np.random.choice(self.dist, p=self.dist_p, size=self.batch_size)
        pairs = [self.all_pairs[d] for d in dists]

        def pick(pair):
            idx = np.random.choice(pair.shape[1])
            return pair[:, idx]

        pairs = np.stack([pick(p) for p in pairs])
        labs = np.random.choice((0, 1), size=self.batch_size)
        rev_idx = labs == 0
        pairs[rev_idx] = np.flip(pairs[rev_idx], axis=1)
        return pairs, labs

    def __iter__(self):
        return self


if __name__ == '__main__':
    task = TiTask(dist=[1,2])
    next(iter(task))

# %%
