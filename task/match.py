"""
Matching tasks, analogous to the delayed match-to-sample task of Griffiths paper (TODO: cite)
"""
# <codecell>
import functools

import jax
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
    def __init__(self, n_points=8, n_classes=128, n_labels=32, n_dims=64,
                 matched_target=True,
                 bursty=1, prob_b=1, 
                 eps=0.1, alpha=0, width=1,
                 batch_size=128, seed=None, reset_rng_for_data=True):
        self.n_points = n_points
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.n_dims = n_dims
        self.matched_target = matched_target
        self.bursty = bursty
        self.prob_b = prob_b
        self.eps = eps
        self.alpha = alpha
        self.width = width
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.width = width

        self.idx_to_label = self.rng.normal(loc=0, scale=(self.width / np.sqrt(n_dims)), size=(self.n_labels, n_dims))
        if n_classes is not None:
            self.idx_to_center = self.rng.normal(loc=0, scale=(self.width / np.sqrt(n_dims)), size=(self.n_classes, n_dims))

            assert n_classes % n_labels == 0, f'n_classes={n_classes} is not divisible by n_labels={n_labels}'
            n_classes_per_lab = n_classes // n_labels
            self.class_to_label = np.repeat(np.arange(self.n_labels), repeats=n_classes_per_lab)
            self.class_to_label = self.rng.permutation(self.class_to_label)
        
        assert self.n_points % self.bursty == 0, f'n_points={self.n_points} is not divisible by bursty={self.bursty}'

        if reset_rng_for_data:
            self.rng = np.random.default_rng(None)
    
    def resample_clusters(self, seed=None):
        if self.n_classes is None:
            return

        rng = np.random.default_rng(seed)
        self.idx_to_center = rng.normal(loc=0, scale=(self.width / np.sqrt(self.n_dims)), size=(self.n_classes, self.n_dims))

    def swap_labels(self, seed=None):
        if self.n_classes is None:
            return

        rng = np.random.default_rng(seed)
        self.class_to_label = rng.permutation(self.class_to_label)

    def _sample_example(self, rng, burst=0):
        if self.n_classes is not None:
            cluster_probs = np.arange(1, self.n_classes + 1)**(-self.alpha)
            cluster_probs = cluster_probs / np.sum(cluster_probs)

            if burst > 0:
                cluster_idxs = rng.choice(self.n_classes, size=self.n_points // burst, p=cluster_probs, replace=False)
                cluster_idxs = np.repeat(cluster_idxs, repeats=burst)
                cluster_idxs = rng.permutation(cluster_idxs)
            else:
                cluster_idxs = rng.choice(self.n_classes, size=self.n_points, p=cluster_probs, replace=True)
            
            if self.matched_target:
                target_idx = rng.choice(cluster_idxs)
            else:
                target_idx = rng.choice(self.n_classes)
                while self.class_to_label[target_idx] in self.class_to_label[cluster_idxs]:
                    target_idx = rng.choice(self.n_classes)

            cluster_idxs = np.append(cluster_idxs, target_idx)
            label_idxs = self.class_to_label[cluster_idxs]
            centers = self.idx_to_center[cluster_idxs]
        else:
            centers = rng.normal(loc=0, scale=(self.width / np.sqrt(self.n_dims)), size=(self.n_points + 1, self.n_dims))

            if burst > 0:
                label_idxs = rng.choice(self.n_labels, size=self.n_points // burst, replace=False)
                label_idxs = np.repeat(label_idxs, repeats=burst)
                label_idxs = rng.permutation(label_idxs)
            else:
                label_idxs = rng.choice(self.n_classes, size=self.n_points, replace=True)

            if self.matched_target:
                target = rng.choice(self.n_points - 1)
                centers[-1] = centers[target]
                label_idxs = np.append(label_idxs, label_idxs[target])
            else:
                target_idx = self.rng.choice(self.n_labels)
                while target_idx in label_idxs:
                    target_idx = self.rng.choice(self.n_labels)

                label_idxs = np.append(label_idxs, target_idx)

        points = (centers + self.eps * rng.normal(scale=(1/np.sqrt(self.n_dims)), size=(centers.shape))) / np.sqrt(1 + self.eps**2)
        labels = self.idx_to_label[label_idxs]

        xs = np.empty((2 * self.n_points + 1, self.n_dims))
        xs[0::2] = points
        xs[1::2] = labels[:-1]
        return xs, label_idxs[-1]
    
    def __next__(self):
        n_bursty = self.rng.binomial(self.batch_size, p=self.prob_b)
        args = np.zeros(self.batch_size, dtype=np.int32)
        args[:n_bursty] = self.bursty
        all_exs = [self._sample_example(self.rng, burst=a) for a in args]

        xs, ys = zip(*all_exs)
        perm_idxs = self.rng.permutation(self.batch_size)
        return np.array(xs)[perm_idxs], np.array(ys)[perm_idxs]

    def __iter__(self):
        return self
    
    def close(self):
        self.pool.close()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_points = 32
    n_bursty = n_points // 4

    task = GautamMatch(matched_target=False, n_points=n_points, n_classes=1024, n_labels=32, batch_size=2, seed=50, eps=0.1, bursty=n_bursty, n_dims=2, reset_rng_for_data=True)
    # print(task.idx_to_label)
    # print(task.class_to_label)
    xs, ys = next(task)

    x = xs[0,:-1:2]
    
    plt.scatter(x[:,0], x[:,1])
    plt.gca().set_axis_off()

    plt.gcf().set_size_inches(2.5, 2.5)
    plt.tight_layout()
    plt.savefig('../experiment/fig/final/fig1/cls_icl_example.svg')


    # task = RingMatch(radius=1, batch_size=10, seed=1, reset_rng_for_data=True)

    # fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.5))

    # for ax, (xs, ys) in zip([axs], task):
    #     c = np.zeros(6)
    #     c[ys[0]] = 0
    #     c[-1] = 1

    #     ax.scatter(xs[0,:,0], xs[0,:,1], c=c)
    #     ax.axis('equal')
    #     ax.set_axis_off()
    #     # plt.colorbar()
    
    # plt.tight_layout()
    # plt.savefig('../experiment/fig/match_example.svg')
# %%
