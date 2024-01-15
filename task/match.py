"""
Matching tasks, analogous to the delayed match-to-sample task of Griffiths paper (TODO: cite)
"""
# <codecell>

import numpy as np

class RingMatch:
    def __init__(self, n_points=6, radius=1, scramble=False, batch_size=128, data_size=None, seed=None, reset_rng_for_data=False) -> None:
        self.n_points = n_points
        self.radius = radius
        self.scramble = scramble
        self.batch_size = batch_size
        self.data_size = data_size
        self.rng = np.random.default_rng(seed)

        if self.data_size is not None:
            self.data = self.sample(self.data_size)
        
        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
        
    
    def sample(self, size):
        start = self.rng.uniform(0, 2 * np.pi, size=size)
        incs = 2 * np.pi / (self.n_points - 1)
        angles = np.array([start + incs * i for i in range(self.n_points - 1)] + [self.rng.uniform(0, 2 * np.pi, size=size)]).T
        
        xs = self.radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)

        if self.scramble:
            list(map(self.rng.shuffle, xs[:,:-1,:]))

        xs_choice = np.transpose(xs[:,[-1],:], axes=(0, 2, 1))
        dots = (xs[:,:-1,:] @ xs_choice).squeeze()
        closest_idxs = np.argmax(dots, axis=1)
        return xs, closest_idxs
    
    def __next__(self):
        if self.data_size is None:
            return self.sample(self.batch_size)
        else:
            idxs = self.rng.choice(self.data_size, size=self.batch_size, replace=True)
            return self.data[0][idxs], self.data[1][idxs]

    def __iter__(self):
        return self


class LabelRingMatch:
    """Gautam's classification task (simplified)"""
    def __init__(self, n_points=4, radius=1, n_classes=None, scramble=True, batch_size=128, seed=None, reset_rng_for_data=True):
        self.n_points = n_points
        self.radius = radius
        self.n_classes = n_classes or (n_points - 1)
        self.scramble = scramble
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.idx_to_label = self.rng.normal(loc=0, scale=(1 / np.sqrt(2)), size=(self.n_classes, 2)) # 2D dataset

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def __next__(self):
        start = self.rng.uniform(0, 2 * np.pi, size=self.batch_size)
        incs = 2 * np.pi / (self.n_points - 1)
        angles = np.array([start + incs * i for i in range(self.n_points - 1)] + [self.rng.uniform(0, 2 * np.pi, size=self.batch_size)]).T
        
        xs = self.radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)

        if self.scramble:
            list(map(self.rng.shuffle, xs[:,:-1,:]))

        xs_choice = np.transpose(xs[:,[-1],:], axes=(0, 2, 1))
        dots = (xs[:,:-1,:] @ xs_choice).squeeze()
        closest_idxs = np.argmax(dots, axis=1)

        classes = np.stack([self.rng.choice(self.n_classes, replace=False, size=(self.n_points - 1)) for _ in range(self.batch_size)])
        labels = self.idx_to_label[classes]
        closest_classes = classes[np.arange(self.batch_size), closest_idxs]

        interl_xs = np.empty((self.batch_size, self.n_points * 2 - 1, 2))
        interl_xs[:, 0::2] = xs
        interl_xs[:, 1::2] = labels

        return interl_xs, closest_classes

    def __iter__(self):
        return self


class GautamMatch:
    """Gautam's classification task (full)"""
    def __init__(self, n_points=8, n_classes=64, n_labels=16, n_dims=2, bursty=1, prob_b=1, eps=0.1, alpha=0, batch_size=128, seed=None):
        self.n_points = n_points
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.n_dims = n_dims
        self.bursty = bursty
        self.prob_b = prob_b
        self.eps = eps
        self.alpha = alpha
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # TODO: assignments happen with same RNG as sampling
        self.idx_to_center = self.rng.normal(loc=0, scale=(1 / np.sqrt(n_dims)), size=(self.n_classes, n_dims))
        self.idx_to_label = self.rng.normal(loc=0, scale=(1 / np.sqrt(n_dims)), size=(self.n_labels, n_dims))

        assert n_classes % n_labels == 0, f'n_classes={n_classes} is not divisible by n_labels={n_labels}'
        n_classes_per_lab = n_classes // n_labels
        self.class_to_label = np.repeat(np.arange(self.n_labels), repeats=n_classes_per_lab)
        self.class_to_label = self.rng.permutation(self.class_to_label)
        
        assert self.n_points % self.bursty == 0, f'n_points={self.n_points} is not divisible by bursty={self.bursty}'
    
    def _sample_example(self, burst=0):
        cluster_probs = np.arange(1, self.n_classes + 1)**(-self.alpha)
        cluster_probs = cluster_probs / np.sum(cluster_probs)

        if burst > 0:
            cluster_idxs = self.rng.choice(self.n_classes, size=self.n_points // burst, p=cluster_probs, replace=False)
            cluster_idxs = np.repeat(cluster_idxs, repeats=burst)
            cluster_idxs = self.rng.permutation(cluster_idxs)
        else:
            cluster_idxs = self.rng.choice(self.n_classes, size=self.n_points, p=cluster_probs, replace=True)
        
        target_idx = self.rng.choice(cluster_idxs)
        cluster_idxs = np.append(cluster_idxs, target_idx)

        centers = self.idx_to_center[cluster_idxs]
        points = (centers + self.eps * self.rng.normal(scale=(1/np.sqrt(self.n_dims)), size=(centers.shape))) / np.sqrt(1 + self.eps**2)
        label_idxs = self.class_to_label[cluster_idxs]
        labels = self.idx_to_label[label_idxs]

        xs = np.empty((2 * self.n_points + 1, self.n_dims))
        xs[0::2] = points
        xs[1::2] = labels[:-1]
        return xs, label_idxs[-1]
    
    def __next__(self):
        n_bursty = self.rng.binomial(self.batch_size, p=self.prob_b)
        bursty_exs = [self._sample_example(burst=self.bursty) for _ in range(n_bursty)]
        plain_exs = [self._sample_example(burst=0) for _ in range(self.batch_size - n_bursty)]
        all_exs = bursty_exs + plain_exs
        xs, ys = zip(*all_exs)

        perm_idxs = self.rng.permutation(self.batch_size)
        return np.array(xs)[perm_idxs], np.array(ys)[perm_idxs]

    def __iter__(self):
        return self


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    task = GautamMatch(n_points=4, n_classes=4, n_labels=2, batch_size=5, seed=50, eps=0, bursty=4)
    xs, ys = next(task)
    print(xs)
    print(ys)

    # task = LabelRingMatch(n_points=6, seed=1, reset_rng_for_data=True)

    # xs, labs = next(task)

    # xs = xs[0]

    # points = xs[0::2]
    # labels = xs[1::2]

    # print('POINTS', points)
    # print("LAB", labels)

    # plt.scatter(points[:,0], points[:,1], c=np.arange(6))
    # plt.axis('equal')

    # print('Label', labs[0])
    # print('Match', task.idx_to_label[labs[0]])
    # print("All", task.idx_to_label)

    # task = RingMatch(radius=1, batch_size=10, seed=1, reset_rng_for_data=True)

    # fig, axs = plt.subplots(2, 3, figsize=(6, 4))

    # for ax, (xs, ys) in zip(axs.ravel(), task):
    #     c = np.zeros(6)
    #     c[ys[0]] = 0.3
    #     c[-1] = 1

    #     ax.scatter(xs[0,:,0], xs[0,:,1], c=c)
    #     ax.axis('equal')
    #     # plt.colorbar()
    
    # plt.tight_layout()
    # plt.savefig('../experiment/fig/match_examples.png')