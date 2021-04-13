import numpy as np

WL = np.zeros((6, 17))
WR = np.zeros((10, 6))
for idx, value in np.ndenumerate(WL):
    WL[idx] = idx[0] + 0.01 * idx[1]

for idx, value in np.ndenumerate(WR):
    WR[idx] = idx[1] + 0.01 * idx[0]

BL = 1 + 0.01 * np.arange(0, 6)[:, np.newaxis]

class Reducer:
    def __init__(self):
        pass

    def fit(self, survivor_list, layer_size):
        self.mask = np.zeros((layer_size, int(layer_size / 2)))
        for col_idx, _ in enumerate(self.mask.T):
            self.mask[survivor_list[col_idx], col_idx] = 1
        return self

    def _transform_array(self, arr, side):
        if side == 'left':
            return (arr.T @ self.mask).T
        elif side == 'right':
            return arr @ self.mask

    def _inverse_transform_array(self, arr, side):
        if side == 'left':
            return (arr.T @ self.mask.T).T
        elif side == 'right':
            return arr @ self.mask.T

    def transform(self, wl, bl, wr):
        return [self._transform_array(arr, side) for arr, side in zip([wl, bl, wr], ['left', 'left', 'right'])]

    def inverse_transform(self, wl, bl, wr):
        return [self._inverse_transform_array(arr, side) for arr, side in zip([wl, bl, wr], ['left', 'left', 'right'])]



layer_size = WL.shape[0]
"""survivors = np.random.choice(layer_size, int(layer_size/2), replace=False)"""
survivor_list = np.array([0, 2, 3])


rdcr = Reducer().fit(survivor_list, layer_size)
_d = lambda x: 10 * np.ones_like(x)

WL_red, BL_red, WR_red = rdcr.transform(WL, BL, WR)
dWL, dBL, dWR = rdcr.inverse_transform(_d(WL_red), _d(BL_red), _d(WR_red))

assert WL.shape == dWL.shape
assert WR.shape == dWR.shape
assert BL.shape == dBL.shape

