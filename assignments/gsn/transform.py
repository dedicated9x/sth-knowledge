import numpy as np

prev = np.zeros((6, 17))
_next = np.zeros((10, 6))
for idx, value in np.ndenumerate(prev):
    prev[idx] = idx[0] + 0.01 * idx[1]

for idx, value in np.ndenumerate(_next):
    _next[idx] = idx[1] + 0.01 * idx[0]

# TODO wylosuj multinomial i test na 0,2,3
# TODO prev - zredukuj i odtwórz

class Reducer:
    def __init__(self):
        pass

    def fit(self, survivor_list, layer_size):
        self.mask = np.zeros((layer_size, int(layer_size / 2)))
        for col_idx, _ in enumerate(mask.T):
            self.mask[survivor_list[col_idx], col_idx] = 1
        return self

    def transform(self, arr):
        return (arr.T @ self.mask).T

    def inverse_transform(self, arr):
        return (arr.T @ self.mask.T).T


layer_size = prev.shape[0]
# survivors = np.random.choice(layer_size, int(layer_size/2), replace=False)
survivor_list = np.array([0, 2, 3])

# mask = np.zeros((layer_size, int(layer_size/2)))
# for col_idx, _ in enumerate(mask.T):
#     mask[survivor_list[col_idx], col_idx] = 1
#
# prev_reduced = (prev.T @ mask).T
# grad = 10 * np.ones_like(prev_reduced)
# grad_exploded = (grad.T @ mask.T).T

rdcr = Reducer().fit(survivor_list, layer_size)
prev_reduced = rdcr.transform(prev)
grad_reduced = 10 * np.ones_like(prev_reduced)
grad = rdcr.inverse_transform(grad_reduced)
