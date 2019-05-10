from gates.gate import Gate
from gates.logf import Logf


class Neuron(Gate):
    """
    Neuron with a logistic function non-linearity, $N(x,y) = logf(ax + by +c)$

    >>> from gates.leaf import Const
    >>> a, x, b, y, c = (Const(v) for v in (2, 0.5, 0, 3, -1))
    >>> n = Neuron(a, x, b, y, c)
    >>> n.forward()
    >>> n.val
    array( 0.5)

    >>> n.grad = 4
    >>> n.backward()
    >>> tuple(v for g in (n, a, x) for v in (g.val, g.grad))
    (neuron()[0.5, 4.0], par()[2.0, 0.5], par()[0.5, 2.0])
    >>> tuple(v for g in (b, y, c) for v in (g.val, g.grad))
    (par()[0.0, 3.0], par()[3.0, 0.0], par()[-1.0, 1.0])
    """
    name = 'neuron'
    arity = 5

    def __init__(self, a, x, b, y, c):
        super().__init__([a, x, b, y, c])

    def forward(self):
        a, x, b, y, c = (in_node.val for in_node in self.igs)
        self.val = Logf.logf(a * x + b * y + c)

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
