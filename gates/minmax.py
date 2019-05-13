from gates.add import Arity2Plus


class Min(Arity2Plus):
    """
    >>> from nn.sugar import *
    >>> a, b, c = param(1, -2, 3)
    >>> tuple(float(x.compute()) for x in (MinGate(a), minimum(a), minimum(a, b), minimum(a, b, c)))
    (1.0, 1.0, -2.0, -2.0)

    >>> min_abc = minimum(a, b, c)
    >>> _ = min_abc.compute()
    >>> min_abc.backprop(grad=0.1)
    >>> tuple(float(x.grad) for x in (a, b, c))
    (0.0, 0.1, 0.0)
    """
    name = 'min'

    def __init__(self, *argv):
        super().__init__(*argv)
        self.order_fun = min

    def forward(self):
        self.val = self.order_fun(gate.val for gate in self.igs)

    def backward(self):
        for gate in self.igs:
            if gate.val == self.val:
                gate.grad += self.grad


class Max(Min):
    """
    >>> from nn.sugar import *
    >>> a, b, c = param(1, -2, 3)
    >>> tuple(float(x.compute()) for x in (MaxGate(a), maximum(a), maximum(a, b), maximum(a, b, c)))
    (1.0, 1.0, 1.0, 3.0)

    >>> max_abc = maximum(a, b, c)
    >>> _ = max_abc.compute()
    >>> max_abc.backprop(grad=0.1)
    >>> tuple(float(x.grad) for x in (a, b, c))
    (0.0, 0.0, 0.1)
    """
    name = 'max'

    def __init__(self, *argv):
        super().__init__(*argv)
        self.order_fun = max


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
