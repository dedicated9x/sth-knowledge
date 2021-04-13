import numpy as np

prev = np.zeros((6, 17))
_next = np.zeros((10, 6))
for idx, value in np.ndenumerate(prev):
    prev[idx] = idx[0] + 0.01 * idx[1]

for idx, value in np.ndenumerate(_next):
    _next[idx] = idx[1] + 0.01 * idx[0]

# TODO prev - zredukuj i odtwórz
prev_b = 1 + 0.01 * np.arange(0, 6)[:, np.newaxis]

class Reducer:
    def __init__(self):
        pass

    def fit(self, survivor_list, layer_size):
        self.mask = np.zeros((layer_size, int(layer_size / 2)))
        for col_idx, _ in enumerate(self.mask.T):
            self.mask[survivor_list[col_idx], col_idx] = 1
        return self

    def transform(self, arr, side):
        if side == 'prev':
            return (arr.T @ self.mask).T
        elif side == 'next':
            return arr @ self.mask

    def inverse_transform(self, arr, side):
        if side == 'prev':
            return (arr.T @ self.mask.T).T
        elif side == 'next':
            return arr @ self.mask.T




layer_size = prev.shape[0]
"""survivors = np.random.choice(layer_size, int(layer_size/2), replace=False)"""
survivor_list = np.array([0, 2, 3])


rdcr = Reducer().fit(survivor_list, layer_size)
prev_reduced = rdcr.transform(prev, side='prev')
grad_prev_reduced = 10 * np.ones_like(prev_reduced)
grad_prev = rdcr.inverse_transform(grad_prev_reduced, side='prev')

next_reduced = rdcr.transform(_next, side='next')
grad_next_reduced = 10 * np.ones_like(next_reduced)
grad_next = rdcr.inverse_transform(grad_next_reduced, side='next')

prev_b_reduced = rdcr.transform(prev_b, side='prev')
grad_prev_b_reduced = 10 * np.ones_like(prev_b_reduced)
grad_prev_b = rdcr.inverse_transform(grad_prev_b_reduced, side='prev')
# grad_prev_b = (grad_prev_b_reduced.T @ rdcr.mask.T).T