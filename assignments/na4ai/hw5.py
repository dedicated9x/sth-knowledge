import numpy as np

X = np.array([1., 1., 1.])[:, np.newaxis]

# np.array([1., 1., 1.])[:, np.newaxis, np.newaxis]

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

lenghts = [h.shape[0] for h in [H1, H2]]

dH2dW2 = [row[:, np.newaxis] @ H1.T for row in np.identity(lenghts[1])]
dH1dW1 = [row[:, np.newaxis] @ X.T for row in np.identity(lenghts[1])]

# TODO te kurewskie przemnożenie