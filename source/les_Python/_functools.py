from functools import partial

multiply = lambda a, b: a + b
duplicate = partial(multiply, b=2)
print(duplicate(3))
