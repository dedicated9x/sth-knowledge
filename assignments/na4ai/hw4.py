import numpy as np
import copy

def partial_derivate(base_func, var_name):
    h = 0.0001
    def wrapper(*args, **params):
        alt_params = params.copy()
        alt_params[var_name] += h
        return (base_func(*args, **alt_params) - base_func(*args, **params)) / h
    return wrapper


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
        return '{:3s} value: {:5.3f}    grad: {:5.3f}'.format(self.id, self.value, self.grad)

    def children_dict(self):
        """
        Helper method.
        """
        return {e.id: e.value for e in self.children}

    def calculate_value(self):
        self.value = self.func(**self.children_dict())

    def calculate_grad(self):
        self.grad = sum([p.grad * partial_derivate(p.func, self.id)(**p.children_dict()) for p in self.parents])


class Backprop:
    def __init__(self, system, direct_func):
        self.system = system
        self.direct_func = direct_func

    def __repr__(self):
        return '\n'.join([e.__repr__() for e in self.system])


    def forward(self):
        _ = [node.calculate_value() for node in self.system if node.type_ != 'input']

    def backward(self):
        _ = [node.calculate_grad() for node in self.system[::-1] if node.type_ != 'output']

    def check(self):
        """
        Compare direct gradient computation result with backpropagation result.
        """
        input_nodes = [node for node in self.system if node.type_ == 'input']
        input_dict = {e.id: e.value for e in input_nodes}
        backprop_grad = [e.grad for e in input_nodes]
        direct_grad = [partial_derivate(self.direct_func, k)(**input_dict) for k in input_dict.keys()]
        print(f'Backpropagation gradient:   {backprop_grad} \n'
              f'Direct gradient:            {direct_grad}')


def sigmoid(x):
    return 1 / (1 + np.exp(-1. * x))

def relu(x):
    return np.maximum(0, x)

"""GRAPH 1 """
# x = Node('x', None, [], type_='input')
# y = Node('y', None, [], type_='input')
# z = Node('z', None, [], type_='input')
# v1 = Node('v1', lambda x, y, z: x * y + z, [x, y, z])
# v2 = Node('v2', lambda x, y, z: x - y * z, [x, y, z])
# v3 = Node('v3', lambda v1: v1 ** 2, [v1])
# v4 = Node('v4', lambda v2: np.exp(v2), [v2])
# v5 = Node('v5', lambda v3, v4: v3 + v4, [v3, v4], type_='output')
#
# bp = Backprop([x, y, z, v1, v2, v3, v4, v5], lambda x, y, z: (x * y + z) ** 2 + np.exp(x - y * z))
#
# x.value, y.value, z.value = 1., 2., 1.
# bp.forward()
# bp.backward()
# bp.check()


"""GRAPH 2 """
# x = Node('x', None, [], type_='input')
# y = Node('y', None, [], type_='input')
# z = Node('z', None, [], type_='input')
# v1 = Node('v1', lambda x, y, z: x * y + x * z, [x, y, z])
# v2 = Node('v2', lambda x, y, z: x * z - y * z, [x, y, z])
# v3 = Node('v3', lambda v1: sigmoid(v1), [v1])
# v4 = Node('v4', lambda v2: np.arctan(v2), [v2])
# v5 = Node('v5', lambda v3, v4: v3 + v4, [v3, v4], type_='output')
#
# bp = Backprop([x, y, z, v1, v2, v3, v4, v5], lambda x, y, z: sigmoid(x * y + x * z) + np.arctan(x * z - y * z))
#
# x.value, y.value, z.value = 1., 1., 1.
# bp.forward()
# bp.backward()
# bp.check()

"""GRAPH 3 """
# x = Node('x', None, [], type_='input')
# y = Node('y', None, [], type_='input')
# v1 = Node('v1', lambda x, y: 2 * x + y, [x, y])
# v2 = Node('v2', lambda x, y: x - 2 * y, [x, y])
# v3 = Node('v3', lambda v1: sigmoid(v1), [v1])
# v4 = Node('v4', lambda v2: sigmoid(v2), [v2])
# v5 = Node('v5', lambda v3, v4: v3 - v4, [v3, v4])
# v6 = Node('v6', lambda v3, v4: v3 + v4, [v3, v4])
# v7 = Node('v7', lambda v5: sigmoid(v5), [v5])
# v8 = Node('v8', lambda v6: sigmoid(v6), [v6])
# v9 = Node('v9', lambda v7, v8: v7 + v8, [v7, v8], type_='output')
#
# bp = Backprop([x, y, v1, v2, v3, v4, v5, v6, v7, v8, v9], lambda x, y: sigmoid(sigmoid(2 * x + y) - sigmoid(x - 2 * y)) + sigmoid(sigmoid(2 * x + y) + sigmoid(x - 2 * y)))
#
# x.value, y.value = 1., 1.
# bp.forward()
# bp.backward()
# bp.check()


"""GRAPH 4 """
# x = Node('x', None, [], type_='input')
# y = Node('y', None, [], type_='input')
# v1 = Node('v1', lambda x, y: 2 * x + y, [x, y])
# v2 = Node('v2', lambda x, y: x - 2 * y, [x, y])
# v3 = Node('v3', lambda v1: relu(v1), [v1])
# v4 = Node('v4', lambda v2: relu(v2), [v2])
# v5 = Node('v5', lambda v3, v4: v3 - v4, [v3, v4])
# v6 = Node('v6', lambda v3, v4: v3 + v4, [v3, v4])
# v7 = Node('v7', lambda v5: relu(v5), [v5])
# v8 = Node('v8', lambda v6: relu(v6), [v6])
# v9 = Node('v9', lambda v7, v8: v7 + v8, [v7, v8], type_='output')
#
# bp = Backprop([x, y, v1, v2, v3, v4, v5, v6, v7, v8, v9], lambda x, y: relu(relu(2 * x + y) - relu(x - 2 * y)) + relu(relu(2 * x + y) + relu(x - 2 * y)))
#
# x.value, y.value = 1., 1.
# bp.forward()
# bp.backward()
# bp.check()


"""LAST """
x = Node('x', None, [], type_='input')
w = Node('w', None, [], type_='input')
b = Node('b', None, [], type_='input')
v1 = Node('v1', lambda x, w: (x.T @ w)[0][0], [x, w])
v2 = Node('v2', lambda v1, b: v1 + b, [v1, b])
v3 = Node('v3', lambda v2: sigmoid(v2), [v2], type_='output')

bp = Backprop([x, w, v1, v2, v3], lambda x, w, b: 1)

np.random.seed(0)
x.value = np.random.normal(size=(1000, 1))
w.value = np.random.normal(size=(1000, 1))
b.value = np.random.normal()

bp.forward()

# HARDCODED Backward step.
v2.grad = sigmoid(v2.value) * (1 - sigmoid(v2.value)) * v3.grad
b.grad = 1 * v2.grad
v1.grad = 1 * v2.grad
x.grad = w.value.T * v1.grad
w.grad = x.value.T * v1.grad

# Print values
print(x.grad.T)
print(w.grad.T)
print(b.grad)


# Correctnes check
h = 0.00001
def shift(vec, idx):
    """
    Helper function.
    Returns vector shifted by [0, 0, ..., 0, h, 0, ..., 0] at given coordinate (idx).
    """
    vec1 = copy.deepcopy(vec)
    vec1[idx] += h
    return vec1

def direct_func(x, w, b):
    return sigmoid((x.T @ w)[0][0] + b)

# Gradient calculation - direct approach
direct_grad_x = [(direct_func(shift(x.value, idx), w.value, b.value) - direct_func(x.value, w.value, b.value)) / h for idx in range(1000)]
direct_grad_w = [(direct_func(x.value, shift(w.value, idx), b.value) - direct_func(x.value, w.value, b.value)) / h for idx in range(1000)]
direct_grad_b = (direct_func(x.value, w.value, b.value + h) - direct_func(x.value, w.value, b.value)) / h

# Reshape
direct_grad_x = np.array(direct_grad_x)[:, np.newaxis]
direct_grad_w = np.array(direct_grad_w)[:, np.newaxis]

# Compare gradients (by divide them by themself)
(x.grad.T / direct_grad_x).mean()
(w.grad.T / direct_grad_w).mean()
(b.grad / direct_grad_b).mean()
