"""
Faster convergence by adding a global multiplier.
Annotated output:

Accuracy at iteration 0: 66.7 [1.00 -2.00 -1.00 1.00]       << initial sol 4/6
Accuracy at iteration 10: 66.7 [0.50 -2.15 -0.91 0.70]
Accuracy at iteration 20: 66.7 [0.43 -2.16 -0.89 0.66]
Accuracy at iteration 30: 83.3 [-0.01 -2.30 -0.69 0.77]     << 5/6 by iteration 30
Accuracy at iteration 500: 83.3 [0.77 -2.89 0.68 0.89]
Accuracy at iteration 1000: 83.3 [0.95 -3.49 0.97 0.92]
[...snip...]
Accuracy at iteration 9500: 83.3 [1.76 -12.12 5.89 1.95]
Accuracy at iteration 10000: 83.3 [1.70 -12.76 6.32 2.07]
Accuracy at iteration 10500: 83.3 [2.50 -13.06 6.95 2.13]
Accuracy at iteration 11000: 100.0 [2.43 -13.41 6.95 2.16]  << 6/6 after ~11000 iterations
Accuracy at iteration 11500: 100.0 [2.43 -13.41 6.95 2.16]  << margin requirement already satisfied, gradient 0
Accuracy at iteration 12000: 100.0 [2.43 -13.41 6.95 2.16]
"""
from random import choice
from utils.sugar import *

dataset = (((1.2, 0.7), +1.0), ((-0.3, 0.5), -1.0), ((-3.0, -1.0), +1.0),
           ((0.1, 1.0), -1.0), ((3.0, 1.1), -1.0), ((2.1, -3.0), +1.0))

a, b, c, m = param(1, -2, -1, 1)  # initial solution
x, y, label = const(0, 0, 0)  # not affected by backprop
f = minimum(1, label * m * (a * x + b * y + c))

for iteration in range(12001):

    if iteration % 500 == 0 or (iteration % 10 == 0 and iteration < 40):
        correct = sum(f.compute() > 0 for (x.val, y.val), label.val in dataset)
        print 'Accuracy at iteration {}: {:.1f} [{:.2f} * ({:.2f}, {:.2f}, {:.2f})]'.format(
            iteration, (100.0 * correct) / len(dataset), m.val, a.val, b.val, c.val)

    (x.val, y.val), label.val = choice(dataset)
    f.compute()
    f.backprop(0.1)
