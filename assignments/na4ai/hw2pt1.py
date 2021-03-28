import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

h = 0.0001
def partial_derivate(base_func, var_name):
    """
    Function, that create partial derivative form given function.
    base_func -> given function
    var_name -> 'x' or 'y'
    """
    global h
    delta_x = np.where(var_name == 'x', h, 0)
    delta_y = np.where(var_name == 'y', h, 0)
    def wrapper(x, y):
        x1 = x + delta_x
        y1 = y + delta_y
        return (base_func(x1, y1) - base_func(x, y)) / h
    return wrapper


def calculate_meshgrid(min_, max_, nrpts):
    """Returns arrays which are ready to apply vectorized functions."""
    x = np.linspace(min_, max_, nrpts + 1)
    y = np.linspace(min_, max_, nrpts + 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    return xx, yy

def calculate_gradients(func_, xx_, yy_):
    gradx = partial_derivate(func_, 'x')(xx_, yy_)
    grady = partial_derivate(func_, 'y')(xx_, yy_)
    return gradx, grady

def func1(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

xx_sparse, yy_sparse = calculate_meshgrid(-5, 5, 20)
xx_dense, yy_dense = calculate_meshgrid(-5, 5, 100)
gradx_sparse, grady_sparse = calculate_gradients(func1, xx_sparse, yy_sparse)
gradx_dense, grady_dense = calculate_gradients(func1, xx_dense, yy_dense)


heatmap = gradx_dense*gradx_dense + grady_dense*grady_dense
heatmap = heatmap / np.max(np.max(heatmap))
# In order to put more emphasis on the lower values we need apply a [0, 1]->[0, 1] bijection on the heat values.
heatmap = heatmap ** 0.25


"""1"""
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.quiver(xx_sparse, yy_sparse, gradx_sparse, grady_sparse)
im = ax2.pcolormesh(xx_dense, yy_dense, heatmap, cmap=plt.get_cmap('YlGnBu'))
fig.colorbar(im, ax=ax2)

def find_minimum(start_p):
    current_p = start_p
    path = []
    h1 = .1
    while True:
        neighborhood = current_p + np.array([
            [0, h1, 0, -h1, 0, h1, -h1, h1, -h1],
            [0, 0, h1, 0, -h1, h1, -h1, -h1, h1]
        ])
        min_idx = np.argmin(func1(*neighborhood))
        if min_idx == 0:
            break
        path.append(current_p)
        current_p = np.vstack(neighborhood[:, min_idx])

    path = pd.DataFrame(data=np.concatenate(path, axis=1).T, columns=['X', 'Y'])
    path['IS_MIN'] = False
    path.iat[-1, 2] = True
    return path

"""2"""
# res = find_minimum(np.vstack(np.array([-4, 4]))) # -2.9, 31



xs = np.linspace(-5., 5., 6)
ys = np.linspace(-5., 5., 6)

paths = [find_minimum(np.array([l]).reshape(2, 1)) for l in itertools.product(xs, ys)]
paths = pd.concat(paths, axis=0)
mins = paths[paths['IS_MIN'] == True]


hmap = func1(xx_dense, yy_dense)
hmap = hmap / np.max(hmap)
# In order to put more emphasis on the lower values we need apply a [0, 1]->[0, 1] bijection on the heat values.
hmap = hmap ** 0.5

"""pt3"""
# BLACK = '#0a0a0a'
# ORANGE = '#dd8453'
# fig, ax = plt.subplots()
# im = ax.pcolormesh(xx_dense, yy_dense, hmap, cmap=plt.get_cmap('YlGnBu'))
# fig.colorbar(im, ax=ax)
# ax.scatter(paths['X'], paths['Y'], marker=".", linewidths=1., color=BLACK)
# ax.scatter(mins['X'], mins['Y'], color=ORANGE)
# TODO dodaj heatmape