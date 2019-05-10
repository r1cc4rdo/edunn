from random import choice, random
from utils.sugar import *

dataset = (((1.2, 0.7), +1.0), ((-0.3, 0.5), -1.0), ((-3.0, -1.0), +1.0),
           ((0.1, 1.0), -1.0), ((3.0, 1.1), -1.0), ((2.1, -3.0), +1.0))


def hn(xx, yy):  # hidden neuron relu(ax + by + c)
    return relu(param() * xx + param() * yy + param())


x, y, label = const(0, 0, 0)  # score = w_0 + sum_1^3 w_i * hn_i(x, y)
score = summation([param()] + [param() * hn(x, y) for _ in range(3)])

for par in score.parameters():  # randomly initialize weights
    par.val = random() - 0.5

for iteration in range(15001):

    if iteration % 500 == 0 or (iteration % 10 == 0 and iteration < 40):
        correct = sum(score.compute() > 0 for (x.val, y.val), label.val in dataset)
        print('Accuracy at iteration {}: {:.1f}'.format(iteration, (100.0 * correct) / len(dataset)))

    (x.val, y.val), label.val = choice(dataset)
    score.compute()
    score.backprop(lr=0.01)
