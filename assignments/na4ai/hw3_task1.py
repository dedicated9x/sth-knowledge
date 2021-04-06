import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt


def calculate_polynomial_value(params_vec, x: float):
    p = params_vec.shape[1] - 1
    return (x ** np.arange(0, p + 1)) @ params_vec.T

calculate_polynomial_value(np.array([4., 5., 6.])[np.newaxis, :], 2.) # 38 = 4*2^0 + 5*2^1 + 6*2^2


def calculate_polynomial_regression_gradient(params_vec, xs, ys):
    N = xs.shape[0]
    p = params_vec.shape[1] - 1

    ys_pred = calculate_polynomial_value(params_vec, xs)
    partial_gradients = []
    for j in np.arange(p + 1):
        dMSE_dMj = (-2 / N) * (xs ** j).T @ (ys - ys_pred)
        partial_gradients.append(dMSE_dMj[0][0])

    gradient = np.array(partial_gradients).reshape(1, p + 1)
    return gradient



df = np.load(pl.Path(rf"C:\Users\devoted\Downloads\assignment22.npy"))
xs = df[:, 0][:, np.newaxis]
ys = df[:, 1][:, np.newaxis]

P = 2
params_vec_0 = np.random.randn(1, P + 1)

# gradient = calculate_polynomial_regression_gradient(params_vec_0, xs, ys)
import copy

params_vec = params_vec_0
alpha = 0.001
log = []
log_grads = []
for i in range(20000):
    gradient = calculate_polynomial_regression_gradient(params_vec, xs, ys)
    params_vec = params_vec - alpha * gradient
    log.append(copy.deepcopy(params_vec))
    log_grads.append(gradient)

print(np.linalg.norm(log_grads[-1]))
# TODO sprawdzic, czy y sie zgadza
ys_pred = calculate_polynomial_value(params_vec, xs)
fig, ax = plt.subplots()
ax.scatter(xs, ys)
ax.scatter(xs, ys_pred)
