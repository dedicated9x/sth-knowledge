def func(x, y):
    return x + y

f = lambda x, y: x + y

import numpy as np


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
v3 = Node('v3', lambda x, y: x * y, [x, y])
v4 = Node('v4', lambda v3: np.sin(v3), [v3])

x.value = 1
y.value = 0.25 * np.pi

v3.calculate_value()
v4.calculate_value()

