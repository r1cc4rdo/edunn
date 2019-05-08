import abc
import numpy as np

from gates.gate import Gate


class Leaf(Gate):
    """
    Superclass for all arity-0 nodes (which are leaves in the spanning tree).
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

    >>> all(map(lambda x: np.isnan(x), (r.val, r.grad, e.val, e.grad, pi.grad)))
    True

    >>> r.alias, np.round(pi.val, 2)
    ('radius', 3.14)
    """

    @abc.abstractmethod
    def __init__(self, name):
        super(Leaf, self).__init__(name, [])

    def forward(self):
        pass

    def backward(self):
        pass


class Const(Leaf):

    def __init__(self, value):
        super(Const, self).__init__('k')
        self.val = value


class Input(Leaf):

    def __init__(self, alias):
        super(Input, self).__init__('in')
        self.alias = alias


class Weight(Leaf):

    def __init__(self, shape=()):
        super(Weight, self).__init__('w')
        self.val = np.full(shape, np.nan)


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
