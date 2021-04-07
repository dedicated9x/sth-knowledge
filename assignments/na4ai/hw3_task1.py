import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt


def polynomial_value(beta, x: float):
    p = beta.shape[1] - 1
    return (x ** np.arange(0, p + 1)) @ beta.T

"""print"""
# calculate_polynomial_value(np.array([4., 5., 6.])[np.newaxis, :], 2.) # 38 = 4*2^0 + 5*2^1 + 6*2^2


class PolynomialRegressionLossFunction:
    def __init__(self, xs, ys):
        self._xs = xs
        self._ys = ys

    def calculate_gradient(self, beta):
        N = xs.shape[0]
        p = beta.shape[1] - 1

        ys_pred = polynomial_value(beta, xs)
        partial_gradients = []
        for j in np.arange(p + 1):
            dMSE_dMj = (-2 / N) * (self._xs ** j).T @ (self._ys - ys_pred)
            partial_gradients.append(dMSE_dMj[0][0])

        gradient = np.array(partial_gradients).reshape(1, p + 1)
        return gradient


def gradient_descent(beta0, gradient_func, alpha, fitness_func=None, fitness_limit=None, break_limit=int(2e7)):
    beta = beta0
    fitness_log = []
    has_converged = False
    for i in range(break_limit):
        beta_grad = gradient_func(beta)
        beta = beta - alpha * beta_grad
        fitness = fitness_func(beta, beta_grad)
        fitness_log.append(fitness)
        if fitness < fitness_limit:
            has_converged = True
            break
    return beta, has_converged, fitness_log






if __name__ == "__main__":
    df = np.load(pl.Path(rf"C:\Users\devoted\Downloads\assignment22.npy"))
    xs = df[:, 0][:, np.newaxis]
    ys = df[:, 1][:, np.newaxis]


    vecs = [gradient_descent(
        beta0=np.random.randn(1, p + 1),
        gradient_func=PolynomialRegressionLossFunction(xs, ys).calculate_gradient,
        alpha=alpha_,
        fitness_func=lambda beta, beta_grad: np.linalg.norm(beta_grad),
        fitness_limit=1e-5
    )[0] for (p, alpha_) in [
        (1, 0.001),
        (2, 0.001),
        # (3, 0.0001),
        # (5, 0.000001)
    ]]




    x_grid = np.linspace(xs.min(), xs.max(), 500)[:, np.newaxis]
    fig, ax = plt.subplots()
    for params_vec in vecs:
        ys_pred = polynomial_value(params_vec, x_grid)
        ax.scatter(x_grid, ys_pred, marker='.', linewidths=1)
    ax.scatter(xs, ys)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


