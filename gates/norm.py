import numpy as np
from gates.gate import Gate


class Norm(Gate):
    """
    >>> import numpy as np
    >>> from nn.sugar import *
    >>> a, b, c = param(1, 1, 1)
    >>> norm_gates = NormGate(a), norm(a), norm(a, b), norm(a, b, c)
    >>> np.allclose([x.compute() for x in norm_gates], [1.0, 1.0, 2**0.5, 3**0.5])
    True

    >>> a.val, b.val, c.val = 3, 4, 12
    >>> n = norm(a, b, c)
    >>> n.compute()
    array( 13.)

    >>> n.backprop(grad=1)
    >>> np.allclose([x.grad for x in (a, b, c)], [x / 13.0 for x in (3, 4, 12)])
    True
    """
    name = 'norm'
    arity = 0

    def __init__(self, g0, *argv):
        super().__init__([g0] + list(argv))

    def forward(self):
        self.val = np.sqrt(np.sum(np.power(gate.val, 2) for gate in self.igs))

    def backward(self):
        for gate in self.igs:
            gate.grad += (gate.val / self.val) * self.grad if abs(self.val) > 0.0 else self.grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
