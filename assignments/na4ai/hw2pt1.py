
import matplotlib.pyplot as plt
import numpy as np

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

def func1(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def calculate_meshgrid(min_, max_, nrpts):
    x = np.linspace(min_, max_, nrpts + 1)
    y = np.linspace(min_, max_, nrpts + 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    return xx, yy

def calculate_gradients(func_, xx_, yy_):
    gradx = partial_derivate(func_, 'x')(xx_, yy_)
    grady = partial_derivate(func_, 'y')(xx_, yy_)
    return gradx, grady
# TODO quivery

""" # quiver 
x = np.linspace(-5., 5., 21)
y = np.linspace(-5., 5., 21)
xx, yy = np.meshgrid(x, y, sparse=True)
gradx = partial_derivate(func1, 'x')(xx, yy)
grady = partial_derivate(func1, 'y')(xx, yy)
"""
""" plot """
xx_sparse, yy_sparse = calculate_meshgrid(-5, 5, 20)
gradx_sparse, grady_sparse = calculate_gradients(func1, xx_sparse, yy_sparse)
fig, ax = plt.subplots()
q = ax.quiver(xx_sparse, yy_sparse, gradx_sparse, grady_sparse)

"""PT2"""
# x = np.linspace(-5., 5., 101)
# y = np.linspace(-5., 5., 101)
# xx, yy = np.meshgrid(x, y, sparse=True)
# gradx = partial_derivate(func1, 'x')(xx, yy)
# grady = partial_derivate(func1, 'y')(xx, yy)
# heatmap = gradx*gradx + grady*grady
# heatmap = heatmap / np.max(np.max(heatmap))
# cmap = plt.get_cmap('YlGnBu')
# fig, ax = plt.subplots()
# im = ax.pcolormesh(xx, yy, heatmap, cmap=cmap )
# fig.colorbar(im, ax=ax)


"""main"""
xx_dense, yy_dense = calculate_meshgrid(-5, 5, 100)
gradx_dense, grady_dense = calculate_gradients(func1, xx_dense, yy_dense)
heatmap = gradx_dense*gradx_dense + grady_dense*grady_dense
heatmap = heatmap / np.max(np.max(heatmap))
cmap = plt.get_cmap('YlGnBu')
fig, ax = plt.subplots()
im = ax.pcolormesh(xx_dense, yy_dense, heatmap, cmap=cmap )
fig.colorbar(im, ax=ax)
