from gate import Gate
from arity1 import SigmoidGate


class NeuronGate(Gate):
    """
    Neuron with a sigmoid non-linearity, $N(x,y) = sigmoid(ax + by +c)$

    >>> from utils.sugar import *
    >>> a, x, b, y, c = param(2, 0.5, 0, 3, -1)
    >>> n = neuron(a, x, b, y, c)
    >>> n.compute()
    0.5

    >>> n.backprop(grad=4)
    >>> n, a, x
    (neuron[0.5, 4.0], par[2.0, 0.5], par[0.5, 2.0])
    >>> b, y, c
    (par[0.0, 3.0], par[3.0, 0.0], par[-1.0, 1.0])
    """
    def __init__(self, a, x, b, y, c):
        super(NeuronGate, self).__init__('neuron', [a, x, b, y, c])

    def forward(self):
        a, x, b, y, c = (in_node.val for in_node in self.igs)
        self.val = SigmoidGate.sigmoid(a * x + b * y + c)

    def backward(self):
        partial_grad = self.val * (1 - self.val) * self.grad
        a, x, b, y, c = self.igs
        a.grad += partial_grad * x.val
        x.grad += partial_grad * a.val
        b.grad += partial_grad * y.val
        y.grad += partial_grad * b.val
        c.grad += partial_grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
