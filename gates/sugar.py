"""
Syntactic sugar for using gates.
"""
from collections import Iterable
from random import randint
from gates import *


def inputs(*args):
    names = list(args) if len(args) else ['i' + str(randint(1e5, 1e6-1))]
    return (Input(name) for name in names)


def params(*args):
    names = list(args) if len(args) else ['i' + str(randint(1e5, 1e6-1))]
    return (Parameter(name) for name in names)


def _g(gate_or_value):  # upgrade scalar to Constant gate, or return unchanged
    return Constant(gate_or_value) if not isinstance(gate_or_value, Gate) else gate_or_value


def relu(u0):
    return ReluGate(_g(u0))


def sigmoid(u0):
    return SigmoidGate(_g(u0))


def summation(u0, *argv):  # sum is a keyword
    return AddGate(_g(u0), *[_g(x) for x in argv])


def prod(u0, *argv):
    return MulGate(_g(u0), *[_g(x) for x in argv])


def norm(u0, *argv):
    return NormGate(_g(u0), *[_g(x) for x in argv])


def minimum(u0, *argv):  # min is a keyword
    return MinGate(_g(u0), *[_g(x) for x in argv])


def maximum(u0, *argv):  # max is a keyword
    return MaxGate(_g(u0), *[_g(x) for x in argv])


def neuron(a, b, c, x, y):
    return NeuronGate(_g(a), _g(b), _g(c), _g(x), _g(y))


Gate.__add__ = lambda self, other: AddGate(_g(self), _g(other))
Gate.__mul__ = lambda self, other: MulGate(_g(self), _g(other))
Gate.__pow__ = lambda self, other: PowGate(_g(self), _g(other))
Gate.__rpow__ = lambda self, other: PowGate(_g(other), _g(self))

"""
Derived gates and operators
"""

Gate.__radd__ = Gate.__add__
Gate.__rmul__ = Gate.__mul__
Gate.__pos__ = lambda self: _g(self)
Gate.__neg__ = lambda self: _g(self) * _g(-1)
Gate.__sub__ = lambda self, other: _g(self) + _g(other) * _g(-1)
Gate.__rsub__ = lambda self, other: _g(other) + _g(self) * _g(-1)
Gate.__div__ = lambda self, other: _g(self) * (_g(other) ** _g(-1))
Gate.__rdiv__ = lambda self, other: _g(other) * (_g(self) ** _g(-1))
Gate.__truediv__ = Gate.__div__
Gate.__rtruediv__ = Gate.__rdiv__


def sqrt(u0):
    return PowGate(_g(u0), _g(0.5))

"""
Blow up if trying to compare gates and not the values they contain.
Python 2.7 can compare ANY type, this leads to errors such as const(1) > 2 === True (-_-)
See https://stackoverflow.com/questions/2384078/why-is-0-true-in-python for details.
"""

Gate.__lt__ = lambda self, other: 0/0  # BOOM!
Gate.__le__ = lambda self, other: 0/0  # KABOOM!
Gate.__gt__ = lambda self, other: 0/0  # BADABOOM!
Gate.__ge__ = lambda self, other: 0/0  # KAPOW!
