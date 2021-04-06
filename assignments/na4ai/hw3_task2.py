
import numpy as np
#function for generating random positive definite matrix
def get_randQ( n):
    M = np.random.randn(n,n)
    Q = M.dot(M.T)
    return Q

DIM = 2500
Q = get_randQ(DIM)