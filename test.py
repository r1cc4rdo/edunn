from random import choice
from sugar import *

dataset = (((1.2, 0.7), +1.0), ((-0.3, 0.5), -1.0), ((-3.0, -1.0), +1.0),
           ((0.1, 1.0), -1.0), ((3.0, 1.1), -1.0), ((2.1, -3.0), +1.0))

a, b, c = param((1, -2, -1))  # initial solution
x, y, label = const((0, 0, 0))  # const not affected by update_parameters()
f = minimum(1, label * (a * x + b * y + c))


def evaluate_training_accuracy():
    total_correct = 0
    for training_example in dataset:
        (x.val, y.val), label.val = training_example
        total_correct += f.compute() > 0
    return float(total_correct) / len(dataset)


for iteration in range(40000):

    (x.val, y.val), label.val = choice(dataset)

    f.compute()
    f.backprop()

    # a.grad += -a.val
    # b.grad += -b.val

    f.update_parameters(0.1)

    if (iteration + 1) % 2000 == 0:
        print 'Accuracy at iteration {}: {:.1f} [{:.2f} {:.2f} {:.2f}]'.format(
            iteration, 100 * evaluate_training_accuracy(), a.val, b.val, c.val)
