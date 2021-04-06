import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import functools


def polynomial_value(beta, x: float):
    p = beta.shape[1] - 1
    return (x ** np.arange(0, p + 1)) @ beta.T

"""print"""
# calculate_polynomial_value(np.array([4., 5., 6.])[np.newaxis, :], 2.) # 38 = 4*2^0 + 5*2^1 + 6*2^2


def calculate_polynomial_regression_gradient(xs, ys, params_vec):
    N = xs.shape[0]
    p = params_vec.shape[1] - 1

    ys_pred = polynomial_value(params_vec, xs)
    partial_gradients = []
    for j in np.arange(p + 1):
        # a = (-2 / N)
        # b = (xs ** j).T
        # c = (ys - ys_pred)
        # d = b @ c
        # dMSE_dMj = a * d
        dMSE_dMj = (-2 / N) * (xs ** j).T @ (ys - ys_pred)
        partial_gradients.append(dMSE_dMj[0][0])

    gradient = np.array(partial_gradients).reshape(1, p + 1)
    return gradient


def minimize_gradient_descent(gradient_func, params_vec_0, alpha):
    params_vec = params_vec_0
    log = []
    for i in range(20000000):
        gradient = gradient_func(params_vec)
        params_vec = params_vec - alpha * gradient
        gradient_norm = np.linalg.norm(gradient)
        log.append(gradient_norm)
        if gradient_norm < 10e-5:
            break
    return params_vec, log


def polynomial_regression_gradient_descent(gradient_func, p, alpha):
    params_vec_0 = np.random.randn(1, p + 1)
    params_vec, _ = minimize_gradient_descent(gradient_func, params_vec_0, alpha)
    return params_vec





df = np.load(pl.Path(rf"C:\Users\devoted\Downloads\assignment22.npy"))
xs = df[:, 0][:, np.newaxis]
ys = df[:, 1][:, np.newaxis]
gradient_func = functools.partial(calculate_polynomial_regression_gradient, xs, ys)

"""MAIN"""
# P = 5
# params_vec_03 = np.random.randn(1, P + 1)
# params_vec_3, log_3 = minimize_gradient_descent(gradient_func, params_vec_03, alpha=0.000001)



"""test"""
def plot_results(vecs_, xs_, ys_):
    x_grid = np.linspace(xs_.min(), xs_.max(), 500)[:, np.newaxis]
    fig, ax = plt.subplots()
    for params_vec in vecs_:
        ys_pred = polynomial_value(params_vec, x_grid)
        ax.scatter(x_grid, ys_pred, marker='.', linewidths=1)
    ax.scatter(xs_, ys_)

vecs = [
    polynomial_regression_gradient_descent(gradient_func, 1, 0.001),
    polynomial_regression_gradient_descent(gradient_func, 2, 0.001),
    # polynomial_regression_gradient_descent(gradient_func, 3, 0.0001),
    # polynomial_regression_gradient_descent(gradient_func, 5, 0.000001)
]
plot_results(vecs, xs, ys)

# TODO sprawdzenie tej funkcji, czy to na pewno macierz 50x50