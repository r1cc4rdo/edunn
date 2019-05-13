import numpy as np

from gates.gate import Gate


class Pow(Gate):
    """
    The exponentiation function can be used to implement division and n-th root extraction gates.
    All domain checks are left to numpy -- we assume that the operation is well-defined for the
    input base and exponent pair.

    >>> from gates import *
    >>> b, e = Const(2.), Const(3.)  # x ** y
    >>> p = Pow(b, e)
    >>> p.forward()
    >>> tuple(g.val for g in (p, b, e))
    (8.0, 2.0, 3.0)

    >>> p.grad, b.grad, e.grad = -0.1, 0, 0
    >>> p.backward()
    >>> tuple(round(g.grad, 4) for g in (p, b, e))
    (-0.1, -1.2, -0.5545)

    >>> b.val, e.val = 4, 0.5  # sqrt(x)
    >>> p.forward()
    >>> p.val
    2.0

    >>> b.val, e.val = 10.0, -1  # 1 / x
    >>> p.forward()
    >>> p.val
    0.1

    >>> p.grad, b.grad, e.grad = 1, 0, 0
    >>> p.backward()  # \frac{dx^{-1}}{dx} = -x^{-2}
    >>> b.val, b.grad
    (10.0, -0.01)

    >>> b.val, e.val = range(1, 4), 2.0
    >>> p.forward()
    >>> p.val
    array([1., 4., 9.])

    >>> p.grad, b.grad, e.grad = 1, 0, 0
    >>> p.backward()
    >>> tuple(np.round(g.grad, 3) for g in (p, b, e))
    (1, array([2., 4., 6.]), array([0.   , 2.773, 9.888]))
    """
    name = 'pow'
    arity = (2, 2)

    def __init__(self, base, exp):
        super().__init__([base, exp])

    def forward(self):
        self.val = np.power(self.igs[0].val, self.igs[1].val)

    def backward(self):
        base, exp = self.igs
        base.grad += exp.val * np.power(base.val, exp.val - 1) * self.grad
        exp.grad += np.log(base.val) * self.val * self.grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
