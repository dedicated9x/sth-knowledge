import numpy as np
import copy

def f(w2, w1, x):
    return w2 @ w1 @ x

X = np.array([1., 1., 1.])[:, np.newaxis]

W1 = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])
W2 = np.array([
    [1., 1., 0.],
    [0., 1., 1.],
    [1., 0., 1.]
])

H1 = W1 @ X
H2 = W2 @ H1

dH2dW2 = [row[:, np.newaxis] @ H1.T for row in np.identity(H2.shape[0])]
dH1dW1 = [row[:, np.newaxis] @ X.T for row in np.identity(H1.shape[0])]
dH2dW1 = [sum([val * arr for val, arr in zip(row, dH1dW1)]) for row in W2]


# TODO zrob absa
# TODO refactor, by bylo dla drugiej wagi


def direct_tensor(param_name, W, H):
    res = np.zeros((H.shape[0], *W.shape))
    for (i, j), _ in np.ndenumerate(W):
        h = 1e-4
        W_h = copy.deepcopy(W1)
        W_h[i, j] += h

        kwargs = {'w1': W1, 'w2': W2, 'x': X}
        kwargs_h = kwargs.copy()
        kwargs_h[param_name] = W_h

        grad_ij = (f(**kwargs_h) - f(**kwargs)) / h
        res[:, i, j] = grad_ij.flatten()
    return res


dH2dW1_direct = direct_tensor('w1', W1, H1)
# dH2dW2_direct = direct_tensor('w2', W2, H2)
res1 = np.max(np.abs(np.array(dH2dW1) - dH2dW1_direct))
print(res1)







"""insert slice to tenser"""
# a = np.zeros((3, 4, 5))
# a[:, 0, 0] = np.array([1, 2, 3])

"""test"""
# temp = [row for row in W2]
# row = temp[0]
# [val * arr for val, arr in zip(row, dH1dW1)]


