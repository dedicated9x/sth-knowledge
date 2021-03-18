import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Create L-matrix"""
connections = {
    'A': [0., 1., 0., 0., 1., 0., 1.],
    'B': [1., 0., 1., 1., 1., 0., 0.],
    'C': [0., 0., 0., 1., 0., 0., 0.],
    'D': [0., 0., 1., 0., 1., 0., 0.],
    'E': [1., 0., 0., 1., 0., 1., 0.],
    'F': [0., 0., 1., 0., 1., 0., 1.],
    'G': [0., 0., 1., 0., 0., 1., 0.],
}

L = pd.DataFrame.from_dict(connections).apply(lambda x: x / x.sum(), axis=0).values


"""Perform algorithm"""
def sample_r0():
    return (np.random.multinomial(100, [1/7.]*7, size=1) / 100).reshape(7, 1)

def pagerank_algo(r0_, L_, no_steps):
    r = r0_
    for i in range(no_steps):
        r = L_ @ r
    return r

r_algo = pagerank_algo(sample_r0(), L, 100)


eig_vals, eig_vecs = np.linalg.eig(L)
eig_vec_1 = eig_vecs[:, 0].real.reshape(7, 1)
r_algo.flatten() / eig_vec_1.flatten()

"""try multiple (xD) r0-s"""
# r0_list = [sample_r0() for i in range(1000)]
# r_list = [pagerank_algo(r0_, L, 100) for r0_ in r0_list]
# r_list_as_arr = np.concatenate(r_list, axis=1)
# fig, axes = plt.subplots(4, 2)
# for i in range(7):
#     ax = axes[divmod(i, 2)]
#     ax.hist(r_list_as_arr[i, :])


