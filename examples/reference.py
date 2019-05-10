"""
--- Reference 0xbeef
[    0] Accuracy:  66.7 [1.0000 -2.0000 -1.0000]
[    1] Accuracy:  50.0 [1.1200 -1.9300 -0.9000]
[    2] Accuracy:  50.0 [1.1200 -1.9300 -0.9000]
[    3] Accuracy:  50.0 [1.2400 -1.8600 -0.8000]
[    4] Accuracy:  50.0 [1.2400 -1.8600 -0.8000]
[    5] Accuracy:  66.7 [0.9400 -1.9600 -0.7000]
[    6] Accuracy:  66.7 [0.6400 -2.0700 -0.8000]
[    7] Accuracy:  66.7 [0.6400 -2.0700 -0.8000]
[    8] Accuracy:  83.3 [0.3400 -2.1700 -0.7000]
[    9] Accuracy:  83.3 [0.0400 -2.2700 -0.6000]

--- Old linear classifier 0xbeef
[    0] Accuracy:  66.7 [1.0000 -2.0000 -1.0000]
[    1] Accuracy:  50.0 [1.1200 -1.9300 -0.9000]
[    2] Accuracy:  50.0 [1.1200 -1.9300 -0.9000]
[    3] Accuracy:  50.0 [1.2400 -1.8600 -0.8000]
[    4] Accuracy:  50.0 [1.2400 -1.8600 -0.8000]
[    5] Accuracy:  66.7 [0.9400 -1.9600 -0.7000]
[    6] Accuracy:  66.7 [0.6400 -2.0700 -0.8000]
[    7] Accuracy:  66.7 [0.6400 -2.0700 -0.8000]
[    8] Accuracy:  83.3 [0.3400 -2.1700 -0.7000]
[    9] Accuracy:  83.3 [0.0400 -2.2700 -0.6000]
"""
from random import choice, seed


def f(ll, aa, xx, bb, yy, cc):
    return min(1.0, ll * (aa * xx + bb * yy + cc))


lr = 0.1
a, b, c = 1.0, -2.0, -1.0
dataset = ((1.2, 0.7, +1.0), (-0.3, 0.5, -1.0), (-3.0, -1.0, +1.0),  # x, y, label
           (0.1, 1.0, -1.0), (3.0, 1.1, -1.0), (2.1, -3.0, +1.0))

seed(0xbeef)
for iteration in range(10):

    correct = sum(f(label, a, x, b, y, c) > 0 for x, y, label in dataset)
    print('[{:5}] Accuracy: {:5.1f} [{:.4f} {:.4f} {:.4f}]'.format(
        iteration, (100.0 * correct) / len(dataset), a, b, c))

    (x, y), label = choice(dataset)
    if f(label, a, x, b, y, c) < 1.0:
        a += lr * label * x
        b += lr * label * y
        c += lr * label
