"""
Linear classifier example.
For regularization == 1.0, does not converge.
Annotated output for regularization == 0.0:

Accuracy at iteration 0: 66.7 [1.00 -2.00 -1.00]        << initial sol 4/6
Accuracy at iteration 10: 66.7 [0.52 -2.14 -0.90]
Accuracy at iteration 20: 83.3 [-0.08 -2.34 -0.70]      << 5/6 by iteration 20
Accuracy at iteration 30: 83.3 [0.28 -2.13 -0.40]
Accuracy at iteration 2500: 83.3 [0.72 -5.28 1.70]
Accuracy at iteration 5000: 83.3 [1.05 -7.45 3.20]
Accuracy at iteration 7500: 83.3 [1.77 -9.57 4.40]
Accuracy at iteration 10000: 83.3 [1.80 -11.77 5.80]
Accuracy at iteration 12500: 83.3 [3.06 -13.85 7.00]
Accuracy at iteration 15000: 83.3 [2.88 -16.39 8.00]
Accuracy at iteration 17500: 100.0 [3.27 -18.57 9.20]   << 6/6 after ~15000 iterations
Accuracy at iteration 20000: 100.0 [3.75 -20.62 10.50]
Accuracy at iteration 22500: 100.0 [4.08 -22.59 11.60]
Accuracy at iteration 25000: 100.0 [4.38 -24.56 12.50]
Accuracy at iteration 27500: 100.0 [4.53 -26.34 13.70]
Accuracy at iteration 30000: 100.0 [5.01 -27.78 14.50]
Accuracy at iteration 32500: 100.0 [5.13 -28.19 14.60]  << margin requirement satisfied, gradient 0
Accuracy at iteration 35000: 100.0 [5.13 -28.19 14.60]
"""
from random import choice
from utils.sugar import *

dataset = (((1.2, 0.7), +1.0), ((-0.3, 0.5), -1.0), ((-3.0, -1.0), +1.0),
           ((0.1, 1.0), -1.0), ((3.0, 1.1), -1.0), ((2.1, -3.0), +1.0))

a, b, c = param(1, -2, -1)  # initial solution
x, y, label = const(0, 0, 0)  # not affected by backprop
f = minimum(1, label * (a * x + b * y + c))
regularization = 0.0

for iteration in range(35001):

    if iteration % 2500 == 0 or (iteration % 10 == 0 and iteration < 40):
        correct = sum(f.compute() > 0 for (x.val, y.val), label.val in dataset)
        print 'Accuracy at iteration {}: {:.1f} [{:.2f} {:.2f} {:.2f}]'.format(
            iteration, (100.0 * correct) / len(dataset), a.val, b.val, c.val)

    (x.val, y.val), label.val = choice(dataset)
    f.compute()
    f.backprop()

    a.grad -= a.val * regularization
    b.grad -= b.val * regularization

    f.update_parameters(0.1)
