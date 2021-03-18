import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

h = 0.0001

def partial_derivate(base_func, var_name):
    global h
    delta_x = np.where(var_name == 'x', h, 0)
    delta_y = np.where(var_name == 'y', h, 0)
    def wrapper(x, y):
        x1 = x + delta_x
        y1 = y + delta_y
        return (base_func(x1, y1) - base_func(x, y)) / h
    return wrapper

def func1(x, y):
    return -0.25 * (x ** 2 + y ** 2) + 2

def func2(x, y):
    return 1 / (1 + np.exp(-x - y))


x = np.linspace(-5., 5., 501)
y = np.linspace(-5., 5., 501)

xx, yy = np.meshgrid(x, y, sparse=True)
z = func1(xx, yy)

arr1x = partial_derivate(func1, 'x')(xx, yy)
arr1y = partial_derivate(func1, 'y')(xx, yy)

arr2x = partial_derivate(func2, 'x')(xx, yy)
arr2y = partial_derivate(func2, 'y')(xx, yy)

def describe_diff(der_manual, der_auto):
    print(pd.Series(der_manual - der_auto).describe())

row_number_y1 = int(50 * (1. + 5.))

def func1_xder_y1(x):
    return -0.5 * x

der1_manual = func1_xder_y1(x)
der1_auto = arr1x[row_number_y1, :]
plt.scatter(x, der1_auto)
plt.scatter(x, der1_manual)
describe_diff(der1_manual, der1_auto)


def func2_xder_y1(x):
    exp_ = np.exp(-x-1)
    return exp_ / (1 + exp_) ** 2

der2_manual = func2_xder_y1(x)
der2_auto = arr2x[row_number_y1, :]
plt.scatter(x, der2_auto)
plt.scatter(x, der2_manual)
describe_diff(der2_manual, der2_auto)

