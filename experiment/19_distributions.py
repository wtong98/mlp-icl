"""Experimenting with the form of different distributions"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np

N = 100_000
d = 4

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

# u_pred = np.random.normal(0, np.sqrt(1/(2*d)), size=N)
u_pred = us
z2_pred = 1 / np.pi + u_pred + u_pred**2 / np.pi + u_pred**4 / (12 * np.pi)
# z2_pred = np.random.normal(1 / np.pi, np.sqrt(1/(4 * d)), size=N)

plt.hist(z2s, bins=50, density=True, alpha=0.5)
plt.hist(z2_pred, bins=50, density=True, alpha=0.5)

# plt.hist(z2, bins=50)


np.mean(z2s - 1.1*z2d)

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