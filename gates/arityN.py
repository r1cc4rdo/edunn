import math
from operator import mul
from gates.gate import Gate


class AddGate(Gate):
    """
    >>> from utils.sugar import *
    >>> a, b, c = param(1, -2, 3)
    >>> tuple(x.compute() for x in (AddGate(a), a + b, a + b + c))
    (1.0, -1.0, 2.0)

    >>> s = a + b + c
    >>> _ = s.compute()
    >>> s.backprop(grad=0.123)
    >>> tuple(x.grad for x in (a, b, c))
    (0.123, 0.123, 0.123)
    """
    def __init__(self, g0, *argv):
        super(AddGate, self).__init__('+', [g0] + list(argv))

    def forward(self):
        self.val = sum(gate.val for gate in self.igs)

    def backward(self):
        for gate in self.igs:
            gate.grad += self.grad


class MulGate(Gate):
    """
    >>> from utils.sugar import *
    >>> a, b, c = param(1, -2, 3)
    >>> tuple(x.compute() for x in (MulGate(a), a * b, a * b * c))
    (1.0, -2.0, -6.0)

    >>> from utils.numerical import isclose
    >>> s = a * b * c
    >>> _ = s.compute()
    >>> s.backprop(grad=0.1)
    >>> isclose((x.grad for x in (a, b, c)), (-0.6, 0.3, -0.2))
    True
    """
    def __init__(self, g0, *argv):
        super(MulGate, self).__init__('*', [g0] + list(argv))

    def forward(self):
        self.val = reduce(mul, (gate.val for gate in self.igs), 1)

    def backward(self):
        for index, gate in enumerate(self.igs):
            all_other_gates = self.igs[:index] + self.igs[index + 1:]
            gate.grad += reduce(mul, (g.val for g in all_other_gates), 1) * self.grad


class NormGate(Gate):
    """
    >>> from utils.sugar import *
    >>> from utils.numerical import isclose
    >>> a, b, c = param(1, 1, 1)
    >>> isclose((x.compute() for x in (NormGate(a), norm(a), norm(a, b), norm(a, b, c))), (1.0, 1.0, 2**0.5, 3**0.5))
    True

    >>> a.val, b.val, c.val = 3, 4, 12
    >>> n = norm(a, b, c)
    >>> n.compute()
    13.0

    >>> n.backprop(grad=1)
    >>> isclose((x.grad for x in (a, b, c)), (x / 13.0 for x in (3, 4, 12)))
    True
    """
    def __init__(self, g0, *argv):
        super(NormGate, self).__init__('norm', [g0] + list(argv))

    def forward(self):
        self.val = math.sqrt(sum(gate.val**2 for gate in self.igs))

    def backward(self):
        for gate in self.igs:
            gate.grad += (gate.val / self.val) * self.grad if abs(self.val) > 0.0 else self.grad


class MinGate(Gate):
    """
    >>> from utils.sugar import *
    >>> a, b, c = param(1, -2, 3)
    >>> tuple(x.compute() for x in (MinGate(a), minimum(a), minimum(a, b), minimum(a, b, c)))
    (1.0, 1.0, -2.0, -2.0)

    >>> min_abc = minimum(a, b, c)
    >>> _ = min_abc.compute()
    >>> min_abc.backprop(grad=0.1)
    >>> tuple(x.grad for x in (a, b, c))
    (0.0, 0.1, 0.0)
    """
    def __init__(self, g0, *argv):
        super(MinGate, self).__init__('min', [g0] + list(argv))
        self.order_fun = min

    def forward(self):
        self.val = self.order_fun(gate.val for gate in self.igs)

    def backward(self):
        for gate in self.igs:
            if gate.val == self.val:
                gate.grad += self.grad


class MaxGate(MinGate):
    """
    >>> from utils.sugar import *
    >>> a, b, c = param(1, -2, 3)
    >>> tuple(x.compute() for x in (MaxGate(a), maximum(a), maximum(a, b), maximum(a, b, c)))
    (1.0, 1.0, 1.0, 3.0)

    >>> max_abc = maximum(a, b, c)
    >>> _ = max_abc.compute()
    >>> max_abc.backprop(grad=0.1)
    >>> tuple(x.grad for x in (a, b, c))
    (0.0, 0.0, 0.1)
    """
    def __init__(self, g0, *argv):
        super(MaxGate, self).__init__(g0, *argv)
        self.name, self.order_fun = 'max', max


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
