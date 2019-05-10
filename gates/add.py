from gates.gate import Gate


class AddGate(Gate):
    """
    >>> from nn.sugar import *
    >>> a, b, c = param(1, -2, 3)
    >>> tuple(float(x.compute()) for x in (AddGate(a), a + b, summation(a, b), a + b + c, summation(a, b, c)))
    (1.0, -1.0, -1.0, 2.0, 2.0)

    >>> s = a + b + c
    >>> _ = s.compute()
    >>> s.backprop(grad=0.123)
    >>> tuple(float(x.grad) for x in (a, b, c))
    (0.123, 0.123, 0.123)
    """
    name = 'add'
    arity = 0

    def __init__(self, g0, *argv):
        super(AddGate, self).__init__('+', [g0] + list(argv))

    def forward(self):
        self.val = sum(gate.val for gate in self.igs)

    def backward(self):
        for gate in self.igs:
            gate.grad += self.grad
