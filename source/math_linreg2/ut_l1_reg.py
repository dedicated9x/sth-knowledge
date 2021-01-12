import numpy as np
from math_linreg2 import l1_reg

class TestName:
    def test_sum_Jth_column_squared(self):
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        assert l1_reg.sum_Jth_column_squared(X, 1) == 214

    def test_split_theta(self):
        theta = np.vstack(np.array([5, 6, 7, 8]))
        theta_J, theta_dash = l1_reg.split_theta(theta, 2)
        assert theta_J == 7
        np.testing.assert_array_equal(theta_dash, np.vstack(np.array([5, 6, 0, 8])))

