from random import choice

dataset = (((1.2, 0.7), +1), ((-0.3, 0.5), -1), ((-3, -1), +1),
           ((0.1, 1.0), -1), ((3.0, 1.1), -1), ((2.1, -3), +1))

a, b, c = param((1.0, 2.0, -3.0))
x, y = const(dataset[0][0])  # constant in the sense that it is not updated by incoming gradients
f = a * x + b * y + c


def evaluate_training_accuracy():
    total_correct = 0
    for training_example in dataset:
        (x.val, y.val), label = training_example
        output = 1 if f.compute() > 0 else -1
        correct = 1 if output == label else 0
        total_correct += correct
    return total_correct / len(dataset)


for iteration in range(400):

    (x.val, y.val), label = choice(dataset)
    output = f.compute()

    f.grad = 0
    if label == 1 and output < 1:
        f.grad = 1
    if label == -1 and output > -1:
        f.grad = -1
    f.backprop()

    for p in f.parameters():
        print(p.val)
        p.grad += p.val
    f.update_parameters(0.01)

    if iteration % 25 == 0:
        print 'Accuracy at iteration {}: {}'.format(iteration, evaluate_training_accuracy())