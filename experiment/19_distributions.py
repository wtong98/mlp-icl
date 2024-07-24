"""Experimenting with the form of different distributions"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


### Decay of expectation is 1/d
N = 10_000
# d = 400
ds = [2, 4, 8, 16, 32, 64, 128, 256, 512]
all_res = []

for d in tqdm(ds):
    xs_diff = np.random.normal(0, 1/np.sqrt(d), size=(N, 2*d))
    xs_same = np.random.normal(0, 1/np.sqrt(d), size=(N, 2*d))
    xs_same[:,d:] = xs_same[:,:d]

    x1s, x2s = xs_same[:N//2], xs_same[N//2:]
    x1d, x2d = xs_diff[:N//2], xs_same[N//2:]

    # z1 = np.diag(x1 @ x2.T)
    z1s = np.einsum('ij,ij->i', x1s, x2s)
    z1d = np.einsum('ij,ij->i', x1d, x2d)

    us = z1s / (np.linalg.norm(x1s, axis=1) * np.linalg.norm(x2s, axis=1))
    ud = z1d / (np.linalg.norm(x1d, axis=1) * np.linalg.norm(x2d, axis=1))

    z2s = z1s * (1 - np.arccos(us) / np.pi) + (1 / np.pi) * np.sqrt(1 - us**2)
    z2d = z1d * (1 - np.arccos(ud) / np.pi) + (1 / np.pi) * np.sqrt(1 - ud**2)

    # u_pred = us
    # z2_pred = 1 / np.pi + u_pred + u_pred**2 / np.pi + u_pred**4 / (12 * np.pi)

    # plt.hist(z2s, bins=50, density=True, alpha=0.5)
    # plt.hist(z2_pred, bins=50, density=True, alpha=0.5)

    # plt.hist(z2, bins=50)

    res = np.mean(z2s - z2d)
    all_res.append(res)

# <codecell>
n_iters = 100

Ns = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16_000]
ds = [2, 4, 8, 16, 32, 64, 128, 256, 512]

all_res = np.zeros((len(Ns), len(ds)))

for _ in tqdm(range(n_iters)):
    for i, N in enumerate(Ns):
        for j, d in enumerate(ds):
            xs_diff = np.random.normal(0, 1/np.sqrt(d), size=(N, 2*d))
            xs_same = np.random.normal(0, 1/np.sqrt(d), size=(N, 2*d))
            xs_same[:,d:] = xs_same[:,:d]

            x1s, x2s = xs_same[:N//2], xs_same[N//2:]
            x1d, x2d = xs_diff[:N//2], xs_same[N//2:]

            # z1 = np.diag(x1 @ x2.T)
            z1s = np.einsum('ij,ij->i', x1s, x2s)
            z1d = np.einsum('ij,ij->i', x1d, x2d)

            us = z1s / (np.linalg.norm(x1s, axis=1) * np.linalg.norm(x2s, axis=1))
            ud = z1d / (np.linalg.norm(x1d, axis=1) * np.linalg.norm(x2d, axis=1))

            z2s = z1s * (1 - np.arccos(us) / np.pi) + (1 / np.pi) * np.sqrt(1 - us**2)
            z2d = z1d * (1 - np.arccos(ud) / np.pi) + (1 / np.pi) * np.sqrt(1 - ud**2)

            res = np.mean(z2s - z2d)
            all_res[i,j] += (res > 0).astype(int)

# <codecell>
probs = all_res / n_iters
# plt.imshow(probs)

trans_points = np.argwhere(np.cumsum(probs > 0.94, axis=0) == 1)
res = [(Ns[p1], ds[p2]) for (p1, p2) in trans_points]
N, d = zip(*res)
plt.loglog(d, N, '--o')
plt.loglog(d, 10 * np.array(d)**1)

# <codecell>
n_iters = 1000

N = 5_000
# ds = [10, 16, 32, 48, 64, 100, 128, 256, 512, 1024]
ds = (2**(np.linspace(4, 5, num=10))).astype(int)
print('DS', ds)

all_res = np.zeros(len(ds))

for _ in tqdm(range(n_iters)):
    for j, d in enumerate(ds):
        xs_diff = np.random.normal(0, 1/np.sqrt(d), size=(N, 2*d))
        xs_same = np.random.normal(0, 1/np.sqrt(d), size=(N, 2*d))
        xs_same[:,d:] = xs_same[:,:d]

        x1s, x2s = xs_same[:N//2], xs_same[N//2:]
        x1d, x2d = xs_diff[:N//2], xs_same[N//2:]

        # z1 = np.diag(x1 @ x2.T)
        z1s = np.einsum('ij,ij->i', x1s, x2s)
        z1d = np.einsum('ij,ij->i', x1d, x2d)

        us = z1s / (np.linalg.norm(x1s, axis=1) * np.linalg.norm(x2s, axis=1))
        ud = z1d / (np.linalg.norm(x1d, axis=1) * np.linalg.norm(x2d, axis=1))

        z2s = z1s * (1 - np.arccos(us) / np.pi) + (1 / np.pi) * np.sqrt(1 - us**2)
        z2d = z1d * (1 - np.arccos(ud) / np.pi) + (1 / np.pi) * np.sqrt(1 - ud**2)

        res = np.mean(z2s - z2d)
        all_res[j] += (res > 0).astype(int)

# <codecell>
probs = all_res / n_iters
neg_lp = - np.log(probs)

plt.loglog(ds, neg_lp, '--o')
# plt.loglog(ds, [-np.log(0.5)] * len(ds))
plt.loglog(ds, 0.00005 * np.array(ds)**2)
plt.loglog(ds, 0.0015 * np.array(ds)**1)

# <codecell>
pred_probs = np.zeros(probs.shape)

for i, N in enumerate(Ns):
    for j, d in enumerate(ds):
        pred_probs[i, j] = 1 - np.exp(- 1 * N / (d**2))

plt.imshow(pred_probs)
# <codecell>
xs = np.arange(2, 512)
plt.loglog(ds, all_res, '--o')
plt.loglog(xs, 0.2/xs)

# %%
d = 1000
N = 1000

a = np.random.beta(a=d, b=d, size=N)
b = np.random.beta(a=d, b=d, size=N)
c = np.random.beta(a=d, b=d, size=N)

u1 = 4 * a - 2
u2 = 2 * b + 2 * c - 2

val = u1 - u2
plt.hist(u1)
np.mean(val**2)

# np.mean(16 * a**2 - 16 * a + 4)
# np.mean(u1**2)

# np.mean(4 * b**2 + 4 * c**2 - 8 * b - 8 * c + 4)
# np.mean(u2**2)


# <codecell>

xs = np.random.normal(0, 1/np.sqrt(d), size=N)
ys = np.arccos(xs) / np.pi
zs = np.sqrt(1 - xs**2)

ys_pred = np.random.normal(1/2, 1/(np.sqrt(d) * np.pi), size=N)

plt.hist(xs, bins=50, density=True)
plt.hist(ys, bins=50, density=True)
plt.hist(ys_pred, bins=50, density=True)
# plt.hist(zs, bins=50, density=True)