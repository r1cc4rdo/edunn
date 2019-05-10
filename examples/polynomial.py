from utils.sugar import *

x = param(1)  # starting point
f = 3 * x ** 2 - 2 * x + 1

for _ in range(16):  # looks for the global minimum at 2/3, not x: f(x)==0
    print('f({:.5}) = {:.5}'.format(x.val, f.compute()))
    f.backprop(grad=-1, lr=0.1)
