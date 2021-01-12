
import numpy as np
import pandas as pd
import itertools




# def calculate_next_mu(X, MU):
#     distances = pd.DataFrame(np.nan, index=[f"X_{i}" for i in range(X.shape[0])], columns=[f"mu_{i}" for i in range(MU.shape[0])])
#     for i, j in itertools.product(range(X.shape[0]), range(MU.shape[0])):
#         distances.iloc[i, j] = np.linalg.norm(X[i, :] - MU[j, :])
#     distances = distances.T
#
#     c_values = distances.transform(lambda x: [i == x.argmin() for i in range(x.__len__())]).astype(int).T
#     sums = c_values.values.T @ X
#     new_MU = (sums.T * c_values.sum().apply(lambda x: 1/x if x != 0 else 0).values).T
#     unchanged = np.vstack((c_values.sum() == 0).astype(int)) * MU
#     new_MU = new_MU + unchanged
#     return new_MU



def calculate_c_values(X, MU):
    distances = pd.DataFrame(np.nan, index=[f"X_{i}" for i in range(X.shape[0])], columns=[f"mu_{i}" for i in range(MU.shape[0])])
    for i, j in itertools.product(range(X.shape[0]), range(MU.shape[0])):
        distances.iloc[i, j] = np.linalg.norm(X[i, :] - MU[j, :])
    distances = distances.T
    c_values = distances.apply(np.argmin, axis=0)
    return c_values, distances


def calculate_MU(old_MU, X, c_values):
    MU = np.copy(old_MU)
    for j in range(old_MU.shape[0]):
        related_Xs = (c_values == j).astype(int)
        no_related_Xs = related_Xs.sum()
        if no_related_Xs > 0:
            MU[j, :] = (related_Xs.values * X.T).T.sum(axis=0) / no_related_Xs
    return MU


def calculate_distortion(X, MU, c_values):
    distortion = 0
    for i in range(X.shape[0]):
        distortion += np.linalg.norm(X[i, :] - MU[c_values[i], :])
    return distortion


X = np.array([
    [1., 2., 3.],
    [1., 3., -2.],
    [0., 2., 4.],
    [6., -3., 2.],
    [4., 2., 1.],
    [3., -3., 3.]
])

MU_0 = np.array([
    [1., -2., 3.],
    [1., 2., -2.],
    [-2., 2., -2.],
    [3., 2., -2.],
])



# c_values, _distances = calculate_c_values(X, MU)
# new_MU = calculate_MU(MU, X, c_values)
# distortion = calculate_distortion(X, new_MU, c_values)

import pathlib as pl
import matplotlib.pyplot as plt

# X = pd.read_csv(pl.Path(__file__).parent.joinpath('data', 'kmeans', 'X.dat'), sep='  ', header=None, engine='python')
X = pd.read_csv(pl.Path(__file__).parent.joinpath('data', 'kmeans', 'X.dat'), sep='  ', header=None, engine='python').values

# plt.scatter(X[:, 0], X[:, 1])
MU_0 = np.array([[-0.7, 0], [0.3, 0.7], [0.3, -0.5], [0.2, 0.4]])


last_distortion = None
distortion = None
MU = MU_0
while True:
    last_distortion = distortion
    c_values, _distances = calculate_c_values(X, MU)
    MU = calculate_MU(MU, X, c_values)
    distortion = calculate_distortion(X, MU, c_values)
    print(distortion)
    if last_distortion is not None and abs(last_distortion - distortion) < 1e-2:
        break

for j in range(MU_0.shape[0]):
    cluster = X[c_values == j]
    plt.scatter(cluster[:, 0], cluster[:, 1])



"""
c_values = distances.transform(lambda x: [i == x.argmin() for i in range(x.__len__())]).astype(int).T
sums = c_values.values.T @ X
new_MU = (sums.T * c_values.sum().apply(lambda x: 1 / x if x != 0 else 0).values).T
unchanged = np.vstack((c_values.sum() == 0).astype(int)) * MU
new_MU = new_MU + unchanged
"""

# res = calculate_next_mu(X, MU)