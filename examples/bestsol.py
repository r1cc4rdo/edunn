"""
converges to 5.00 -27.50 14.25

"""
from gates.sugar import *

dataset = (((1.2, 0.7), +1.0), ((-0.3, 0.5), -1.0), ((-3.0, -1.0), +1.0),
           ((0.1, 1.0), -1.0), ((3.0, 1.1), -1.0), ((2.1, -3.0), +1.0))

a, b, c, m = params(1, -2, -1, 1)  # initial solution
x, y, label = consts(0, 0, 0)  # not affected by backprop
# f = minimum(1, label * m * (a * x + b * y + c))
f = minimum(1, label * (a * x + b * y + c))
k = norm(a, b, c)  # konstraint

for iteration in range(60001):

    if iteration % 500 == 0 or (iteration % 10 == 0 and iteration < 40):
        correct = sum(f.compute() > 0 for (x.val, y.val), label.val in dataset)
        print('Accuracy at iteration {}: {:.1f} [{:.2f} {:.2f} {:.2f} * {:.2f}]'.format(
            iteration, (100.0 * correct) / len(dataset), a.val, b.val, c.val, m.val))

    ga = gb = gc = gm = 0
    for (x.val, y.val), label.val in dataset:
        # kout = k.compute()
        fout = f.compute()
        f.backprop(grad=min(1, 1 - fout))
        # k.backprop(grad=0.001 * (1 - kout))
        ga += a.grad
        gb += b.grad
        gc += c.grad
        gm += m.grad
    a.grad, b.grad, c.grad, m.grad = ga, gb, gc, gm

    f.update_parameters(lr=0.1)
