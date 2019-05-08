import numpy as np

from gates.gate import Gate


class Pow(Gate):
    """
    The exponentiation function can be used to implement division and n-th root extraction gates.
    All domain checks are left to numpy -- we assume that the operation is well-defined for the
    input base and exponent pair.

    >>> from gates import *
    >>> b, e = Const(2.), Const(3.)
    >>> p = Pow(b, e)
    >>> p.forward()
    >>> p.val
    array( 8.)

    >>> p.grad = -0.1
    >>> p.backward()
    >>> (v for g in (p, b, e) for v in (g.val, g.grad))
    (8.0, -0.1, 2.0, -1.2, 3.0, -0.554517744448)

    >>> b.val, e.val = 4, 0.5  # sqrt(x)
    >>> p.forward()
    >>> p.val
    array( 2.)

    >>> b.val, e.val = 10, -1  # 1 / x
    >>> p.forward()
    >>> p.val
    array( 0.1)

    >>> p.grad = 1
    >>> p.backward()  # \frac{dx^{-1}}{dx} = -x^{-2}
    >>> b.val, b.grad
    (10.0, -0.01)

    >>> tests on arrays
    """
    def __init__(self, g0, g1):
        super(Pow, self).__init__('pow', [g0, g1])

    def forward(self):
        self.val = np.power(self.igs[0].val, self.igs[1].val)

    def backward(self):
        base, exp = self.igs
        base.grad += exp.val * np.power(base.val, exp.val - 1) * self.grad
        exp.grad += np.log(base.val) * self.val * self.grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
