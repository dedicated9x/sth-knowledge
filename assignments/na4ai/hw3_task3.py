import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt

A = np.load(pl.Path(rf"C:\Users\devoted\Downloads\assignment31_A.npy"))
b = np.load(pl.Path(rf"C:\Users\devoted\Downloads\assignment31_b.npy"))


class SoftTreshholdingOperator:
    def __init__(self, p, t):
        self._p = p
        self._t = t

    def T_pt(self, x):
        return np.sign(x) * np.maximum(np.abs(x) - self._p * self._t, 0)



eig_vals, eig_vecs = np.linalg.eig(A.T @ A)
t = 1 / max([e for e in eig_vals if e.imag == 0]).real
p = 1

T_pt = SoftTreshholdingOperator(p, t).T_pt
loss_function = lambda x: 0.5 * np.linalg.norm(A @ x - b) ** 2 + p * np.linalg.norm(x, ord=1)

x0 = np.random.randn(A.shape[1])[:, np.newaxis]

x = x0
loss_log = []
for i in range(50000):
    x = T_pt(x - t * A.T @ (A @ x - b))
    loss_log.append(loss_function(x))

fig, ax = plt.subplots()
ax.scatter(range(len(loss_log)), loss_log)
ax.set_xlabel('no iteration')
ax.set_ylabel('loss function')


