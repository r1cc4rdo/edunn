import numpy as np

from gates.gate import Gate


class Relu(Gate):
    """
    A (Re)ctified (L)inear (U)nit.
    Behaves like $f(x) = x$ if $x \geq 0$, $x = 0$ otherwise.
    Let's compute the value and gradient of relu(x) for x in -2..2:

    >>> from gates import *
    >>> x = Const(np.arange(-2, 3.0))
    >>> r = Relu(x)
    >>> r.forward()
    >>> r.val
    array([0., 0., 0., 1., 2.])

    >>> x.grad = np.zeros_like(x.val)
    >>> r.grad = 0.1 * np.ones_like(x.val)
    >>> r.backward()
    >>> x.grad
    array([0. , 0. , 0. , 0.1, 0.1])
    """
    name = 'relu'
    arity = 1

    def __init__(self, input_gate):
        super().__init__([input_gate])

    def forward(self):
        self.val = np.maximum(self.igs[0].val, 0.0)

    def backward(self):
        self.igs[0].grad += (self.igs[0].val > 0.0) * self.grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
