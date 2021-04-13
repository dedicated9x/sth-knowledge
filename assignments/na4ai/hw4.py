


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

import numpy as np

def partial_derivate(base_func, var_name):
    h = 0.0001
    def wrapper(*args, **params):
        alt_params = params.copy()
        alt_params[var_name] += h
        return (base_func(*args, **alt_params) - base_func(*args, **params)) / h
    return wrapper


# g = partial_derivate(lambda v1, v2: 10 * v1 + v2, 'v1')
# g(v1=2, v2=3)


class Node:
    def __init__(self, id, func, children, type_=None):
        self.id = id
        self.func = func
        self.children = children
        for ch in children:
            ch.parents.append(self)
        self.parents = []
        self.value = None
        self.grad = None
        self.type_ = type_

    def __repr__(self):
        return f'{self.id} - ({self.value}, {self.grad})'

    def children_dict(self):
        return {e.id: e.value for e in self.children}

    def calculate_value(self):
        self.value = self.func(**self.children_dict())
        # return 1

    def calculate_grad(self):
        self.grad = sum([p.grad * partial_derivate(p.func, self.id)(**p.children_dict()) for p in self.parents])

x = Node('x', None, [], type_='input')
y = Node('y', None, [], type_='input')
z = Node('z', None, [], type_='input')
v1 = Node('v1', lambda x, y, z: x * y + z, [x, y, z])
v2 = Node('v2', lambda x, y, z: x - y * z, [x, y, z])
v3 = Node('v3', lambda v1: v1 ** 2, [v1])
v4 = Node('v4', lambda v2: np.exp(v2), [v2])
v5 = Node('v5', lambda v3, v4: v3 + v4, [v3, v4], type_='output')

system = [x, y, z, v1, v2, v3, v4, v5]

x.value = 1.
y.value = 2.
z.value = 0.
_ = [node.calculate_value() for node in system if node.type_ != 'input']
v5.grad = 1
_ = [node.calculate_grad() for node in system[::-1] if node.type_ != 'output']


# TODO direct grad


