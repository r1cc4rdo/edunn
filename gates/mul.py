from gates.gate import Gate


class MulGate(Gate):
    """
    >>> from nn.sugar import *
    >>> a, b, c = param(1, -2, 3)
    >>> tuple(float(x.compute()) for x in (MulGate(a), a * b, prod(a, b), a * b * c, prod(a, b, c)))
    (1.0, -2.0, -2.0, -6.0, -6.0)

    >>> import numpy as np
    >>> s = a * b * c
    >>> _ = s.compute()
    >>> s.backprop(grad=0.1)
    >>> np.allclose([x.grad for x in (a, b, c)], [-0.6, 0.3, -0.2])
    True
    """
    name = 'mul'
    arity = 0

    def __init__(self, g0, *argv):
        super(MulGate, self).__init__('*', [g0] + list(argv))

    def forward(self):
        self.val = reduce(mul, (gate.val for gate in self.igs), 1)

    def backward(self):
        for index, gate in enumerate(self.igs):
            all_other_gates = self.igs[:index] + self.igs[index + 1:]
            gate.grad += reduce(mul, (g.val for g in all_other_gates), 1) * self.grad
