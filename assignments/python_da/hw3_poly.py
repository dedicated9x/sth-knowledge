def sum_poly(A, B):
    if len(A) < len(B):
        A, B = B, A
    B += [0] * (len(A) - len(B))
    res = [a + b for a, b in zip(A, B)]
    while len(res) >= 2 and res[-1] == 0:
        res.pop()
    return res

assert sum_poly([1, 0, 2], [1, 0, -2]) == [2]
assert sum_poly([1, 0, 2], [-1, 0, -2]) == [0]
assert sum_poly([1, 2], [2, 3, 4]) == [3, 5, 4]
assert sum_poly([2, 3, 4], [1, 2]) == [3, 5, 4]