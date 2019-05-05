import math
from gates.gate import Gate


class PowGate(Gate):
    """
    The exponentiation function can be used to implement division and n-th root extraction gates.
    All domain checks are left to the standard library -- we assume that the operation is well-defined
    for the input base and exponent pair.

    >>> from nn.sugar import *
    >>> b, e = param(2, 3)
    >>> p = b ** e
    >>> p.compute()
    array( 8.)

    >>> p.backprop(grad=-0.1)
    >>> p, b, e
    (pow()[8.0, -0.1], par()[2.0, -1.2], par()[3.0, -0.554517744448])

    >>> b.val, e.val = 4, 0.5  # sqrt(x)
    >>> p.compute()
    array( 2.)

    >>> b.val, e.val = 10, -1  # 1 / x
    >>> p.compute()
    array( 0.1)

    >>> p.backprop(grad=1)  # \frac{dx^{-1}}{dx} = -x^{-2}
    >>> b
    par()[10.0, -0.01]
    """
    def __init__(self, g0, g1):
        super(PowGate, self).__init__('pow', [g0, g1])

    def forward(self):
        self.val = math.pow(self.igs[0].val, self.igs[1].val)

    def backward(self):
        base, exp = self.igs
        base.grad += exp.val * math.pow(base.val, exp.val - 1) * self.grad
        exp.grad += math.log(base.val) * self.val * self.grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
