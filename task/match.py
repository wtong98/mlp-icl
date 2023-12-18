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

# <codecell>
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    task = RingMatch(radius=2, scramble=True)
    xs, labs = next(task)
    plt.scatter(xs[0][:,0], xs[0][:,1], c=[0, 1, 2, 3, 4, 5])
    plt.axis('equal')
    labs[0]