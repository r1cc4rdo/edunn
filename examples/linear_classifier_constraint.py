"""
Ask the norm(a, b, c) == 1

"""
from random import choice
from utils.sugar import *

dataset = (((1.2, 0.7), +1.0), ((-0.3, 0.5), -1.0), ((-3.0, -1.0), +1.0),
           ((0.1, 1.0), -1.0), ((3.0, 1.1), -1.0), ((2.1, -3.0), +1.0))

a, b, c, m = param(1, -2, -1, 1)  # initial solution
x, y, label = const(0, 0, 0)  # not affected by backprop
f = minimum(1, label * m * (a * x + b * y + c))
k = norm(a, b, c)  # konstraint

for iteration in range(30001):

    if iteration % 500 == 0 or (iteration % 10 == 0 and iteration < 40):
        correct = sum(f.compute() > 0 for (x.val, y.val), label.val in dataset)
        print 'Accuracy at iteration {}: {:.1f} [{:.2f} {:.2f} {:.2f} * {:.2f}]'.format(
            iteration, (100.0 * correct) / len(dataset), a.val, b.val, c.val, m.val)

    (x.val, y.val), label.val = choice(dataset)

    fout = f.compute()
    kout = k.compute()  # this zeros gradients on a, b, c
    k.backprop(grad=0.001 * (1 - kout))
    f.backprop(grad=min(1, 1 - fout))

    f.update_parameters(lr=0.01)

    # --- converges to param(0.24, -1.34, 0.70, 21.69) in millions of iterations

    # f.compute()
    # kout = k.compute()  # this zeros gradients on a, b, c
    # k.backprop(grad=0.01 * (1 - kout))
    # f.backprop(grad=0.1)
    #
    # f.update_parameters(lr=0.01)
