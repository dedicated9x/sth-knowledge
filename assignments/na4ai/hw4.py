def func(x, y):
    return x + y

f = lambda x, y: x + y

import numpy as np


# TODO type="input", type="output"
"""
x = Node('x', None, [])
y = Node('y', None, [])
v3 = Node('v3', lambda x, y: x * y, [x, y])
v4 = Node('v4', lambda v3: np.sin(v3), [v3])

x.value = 1
y.value = 0.25 * np.pi

v3.calculate_value()
v4.calculate_value()
"""


class Node:
    def __init__(self, id, func, children):
        self.id = id
        self.func = func
        self.children = children
        self.parents = []
        self.value = None
        self.grad = None

    def __repr__(self):
        return f'{self.value} || {self.grad}'

    def calculate_value(self):
        self.value = self.func(**{e.id: e.value for e in self.children})


x = Node('x', None, [])
y = Node('y', None, [])
z = Node('z', None, [])
v1 = Node('v1', lambda x, y, z: x * y - z, [x, y, z])
v2 = Node('v2', lambda x, y, z: x - y * z, [x, y, z])
v3 = Node('v3', lambda v1: v1 ** 2, [v1])
v4 = Node('v4', lambda v2: np.exp(v2), [v2])
v5 = Node('v5', lambda v3, v4: v3 + v4, [v3, v4])

x.value = 1.
y.value = 2.
z.value = 0.

v1.calculate_value()
v2.calculate_value()
v3.calculate_value()
v4.calculate_value()
v5.calculate_value()

