from assignments.na4ai.hw3_task1 import gradient_descent
import matplotlib.pyplot as plt

import numpy as np
#function for generating random positive definite matrix
def get_randQ( n):
    M = np.random.randn(n,n)
    Q = M.dot(M.T)
    return Q

DIM = 2500
Q = get_randQ(DIM)


beta, has_converged, fitness_log = gradient_descent(
    beta0=np.random.randn(DIM)[:, np.newaxis],
    gradient_func=lambda beta: (Q + Q.T) @ beta,
    alpha=0.0001,
    fitness_func=lambda beta, beta_grad: (beta.T @ Q @ beta)[0][0],
    fitness_limit=50,
    break_limit=int(2e3)
)

print(fitness_log[-1])
plt.scatter(range(len(fitness_log)), fitness_log)