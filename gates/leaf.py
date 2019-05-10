import numpy as np

from gates.gate import Gate


class Leaf(Gate):
    """
    Abstract superclass for all arity-0 nodes (which are leaves in the spanning tree).
    No input gates, so nothing to do in forward and backward passes.

    Leaf nodes can be constants, inputs or weights.
    Constants are immutable values, unaffected by gradients reaching them.
    Inputs to a network are named nodes, from where training samples are fed.
    Weights are the floating model parameters, optimized during training.

    For example, let's say we are trying to learn the formula for the area of a
    circle. Given a radius $r$, the area $A$ is equal to: $$A = pi * r ^ e$$ where
    the exponent $e := 2$. In this scenario, pi is a constant, r is an input, e is
    a model parameter to be learned.

    >>> pi = Const(np.pi)
    >>> r = Input('radius')
    >>> e = Weight()

    >>> tuple(x.type for x in (pi, r, e))
    ('const', 'input', 'weight')

    >>> all(map(lambda x: x is None, (r.val, r.grad, e.grad, pi.grad)))
    True

    >>> r.alias, np.round(pi.val, 2), e.val
    ('radius', 3.14, array(nan))
    """
    arity = 0

    def __init__(self):
        super().__init__([])

    def forward(self):
        pass

    def backward(self):
        pass


class Const(Leaf):

    name = 'const'

    def __init__(self, value):
        super().__init__()
        self.val = value


class Input(Leaf):

    name = 'input'

    def __init__(self, alias):
        super().__init__()
        self.alias = alias


class Weight(Leaf):

    name = 'weight'

    def __init__(self, shape=()):
        super().__init__()
        self.val = np.full(shape, np.nan)


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
