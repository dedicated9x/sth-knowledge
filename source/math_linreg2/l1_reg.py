import numpy as np
import pathlib as pl
import pandas as pd

def sum_Jth_column_squared(X, J):
    return sum(X[:, J] ** 2)

def split_theta(theta, J):
    theta_J = theta[J, 0]
    theta_dash = np.copy(theta)
    theta_dash[J, 0] = 0
    return theta_J, theta_dash


def calculate_new_theta_J(X, theta, y, lambda_, J):
    _, theta_dash = split_theta(theta, J)
    a = 0.5 * sum_Jth_column_squared(X, J)
    b = X[:, J] @ (X @ theta_dash - y) # 2469

    common_denominator = 2 * a
    left_parable_vertex = (-b + lambda_) / common_denominator
    right_parable_vertex = (-b - lambda_) / common_denominator

    if left_parable_vertex < 0:
        new_theta_J = left_parable_vertex[0]
    elif right_parable_vertex > 0:
        new_theta_J = right_parable_vertex[0]
    else:
        new_theta_J = 0

    return new_theta_J

def coordinate_ascent(X, theta, y, lambda_):
    last_theta = np.copy(theta)
    while True:
        np.copyto(last_theta, theta)
        for J in range(theta.shape[0]):
            theta[J, 0] = calculate_new_theta_J(X, theta, y, lambda_, J)
        if np.linalg.norm(last_theta - theta) < 1e-05:
            break
    return theta


path_to_data = pl.Path(__file__).parent.joinpath('data', 'l1_reg')
X = pd.read_csv(path_to_data.joinpath('x.dat'), sep='  ', header=None).values
y = pd.read_csv(path_to_data.joinpath('y.dat'), header=None).values

res_df = pd.DataFrame()
for lambda_ in np.logspace(-3, 2, 6):
    print("elo")
    res = coordinate_ascent(X, np.vstack(np.random.normal(size=X.shape[1])), y, lambda_)
    res_df = res_df.assign(**{f"{lambda_}": res.flatten()})
