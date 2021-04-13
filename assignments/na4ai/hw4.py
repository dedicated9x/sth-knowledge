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
        if type_ == 'output':
            self.grad = 1.
        self.type_ = type_

    def __repr__(self):
        return f'{self.id} - ({self.value}, {self.grad})'

    def children_dict(self):
        return {e.id: e.value for e in self.children}

    def calculate_value(self):
        self.value = self.func(**self.children_dict())

    def calculate_grad(self):
        self.grad = sum([p.grad * partial_derivate(p.func, self.id)(**p.children_dict()) for p in self.parents])


class Backprop:
    def __init__(self, system, direct_func):
        self.system = system
        self.direct_func = direct_func

    def forward(self):
        _ = [node.calculate_value() for node in self.system if node.type_ != 'input']

    def backward(self):
        _ = [node.calculate_grad() for node in self.system[::-1] if node.type_ != 'output']

    def check(self):
        input_nodes = [node for node in self.system if node.type_ == 'input']
        input_dict = {e.id: e.value for e in input_nodes}
        backprop_grad = [e.grad for e in input_nodes]
        direct_grad = [partial_derivate(self.direct_func, k)(**input_dict) for k in input_dict.keys()]
        print(f'Backpropagation gradient:   {backprop_grad} \n'
              f'Direct gradient:            {direct_grad}')

x = Node('x', None, [], type_='input')
y = Node('y', None, [], type_='input')
z = Node('z', None, [], type_='input')
v1 = Node('v1', lambda x, y, z: x * y + z, [x, y, z])
v2 = Node('v2', lambda x, y, z: x - y * z, [x, y, z])
v3 = Node('v3', lambda v1: v1 ** 2, [v1])
v4 = Node('v4', lambda v2: np.exp(v2), [v2])
v5 = Node('v5', lambda v3, v4: v3 + v4, [v3, v4], type_='output')

bp = Backprop([x, y, z, v1, v2, v3, v4, v5], lambda x, y, z: (x * y + z) ** 2 + np.exp(x - y * z))

x.value, y.value, z.value = 1., 2., 0.
bp.forward()
bp.backward()
bp.check()



# TODO wklepanie tego plebsu


