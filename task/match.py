"""
Matching tasks, analogous to the delayed match-to-sample task of Griffiths paper (TODO: cite)
"""
# <codecell>

import numpy as np

class RingMatch:
    def __init__(self, n_points=6, radius=1, scramble=False, batch_size=128) -> None:
        self.n_points = n_points
        self.radius = radius
        self.scramble = scramble
        self.batch_size = batch_size
    
    def __next__(self):
        start = np.random.uniform(0, 2 * np.pi, size=self.batch_size)
        incs = 2 * np.pi / (self.n_points - 1)
        angles = np.array([start + incs * i for i in range(self.n_points - 1)] + [np.random.uniform(0, 2 * np.pi, size=self.batch_size)]).T
        
        xs = self.radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)

        if self.scramble:
            list(map(np.random.shuffle, xs[:,:-1,:]))

        xs_choice = np.transpose(xs[:,[-1],:], axes=(0, 2, 1))
        dots = (xs[:,:-1,:] @ xs_choice).squeeze()
        closest_idxs = np.argmax(dots, axis=1)
        return xs, closest_idxs

    def __iter__(self):
        return self


class LabelRingMatch:
    """Gautam's classification task (simplified)"""
    def __init__(self, n_points=4, radius=1, n_classes=None, scramble=True, batch_size=128):
        self.n_points = n_points
        self.radius = radius
        self.n_classes = n_classes or (n_points - 1)
        self.scramble = scramble
        self.batch_size = batch_size

        self.idx_to_label = np.random.randn(self.n_classes, 2) / np.sqrt(2)  # 2D dataset
    
    def __next__(self):
        start = np.random.uniform(0, 2 * np.pi, size=self.batch_size)
        incs = 2 * np.pi / (self.n_points - 1)
        angles = np.array([start + incs * i for i in range(self.n_points - 1)] + [np.random.uniform(0, 2 * np.pi, size=self.batch_size)]).T
        
        xs = self.radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)

        if self.scramble:
            list(map(np.random.shuffle, xs[:,:-1,:]))

        xs_choice = np.transpose(xs[:,[-1],:], axes=(0, 2, 1))
        dots = (xs[:,:-1,:] @ xs_choice).squeeze()
        closest_idxs = np.argmax(dots, axis=1)

        classes = np.stack([np.random.choice(self.n_classes, replace=False, size=(self.n_points - 1)) for _ in range(self.batch_size)])
        labels = self.idx_to_label[classes]
        closest_classes = classes[np.arange(self.batch_size), closest_idxs]

        interl_xs = np.empty((self.batch_size, self.n_points * 2 - 1, 2))
        interl_xs[:, 0::2] = xs
        interl_xs[:, 1::2] = labels

        return interl_xs, closest_classes

    
    def __iter__(self):
        return self

# <codecell>
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    task = LabelRingMatch(n_points=6)

    # <codecell>
    xs, labs = next(task)

    xs = xs[0]

    points = xs[0::2]
    labels = xs[1::2]

    print('POINTS', points)
    print("LAB", labels)

    plt.scatter(points[:,0], points[:,1], c=np.arange(6))
    plt.axis('equal')

    print('Label', labs[0])
    print('Match', task.idx_to_label[labs[0]])
    print("All", task.idx_to_label)

    # task = RingMatch(radius=2, scramble=True)
    # xs, labs = next(task)
    # plt.scatter(xs[0][:,0], xs[0][:,1], c=[0, 1, 2, 3, 4, 5])
    # plt.axis('equal')
    # labs[0]