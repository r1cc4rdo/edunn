"""
Syntactic sugar for using gates.
"""
from collections import Iterable

from gates import *
from gates.neuron import NeuronGate


def const(values=0):
    if not isinstance(values, Iterable):
        return Constant(values)
    return (Constant(v) for v in values)


def param(values=0, floating=True):
    if not isinstance(values, Iterable):
        return Parameter(values, floating)
    return (Parameter(v, floating) for v in values)


def _g(gate_or_value):  # upgrade scalar to Constant gate, or return unchanged
    return Constant(gate_or_value) if not isinstance(gate_or_value, Gate) else gate_or_value


def relu(u0):
    return ReluGate(_g(u0))


def sigmoid(u0):
    return SigmoidGate(_g(u0))


def norm(u0, *argv):
    return NormGate(_g(u0), *[_g(x) for x in argv])


def minimum(u0, *argv):
    return MinGate(_g(u0), *[_g(x) for x in argv])


def maximum(u0, *argv):
    return MaxGate(_g(u0), *[_g(x) for x in argv])


def neuron(a, b, c, x, y):
    return NeuronGate(_g(a), _g(b), _g(c), _g(x), _g(y))


Gate.__add__ = lambda self, other: AddGate(_g(self), _g(other))
Gate.__mul__ = lambda self, other: MulGate(_g(self), _g(other))
Gate.__pow__ = lambda self, other: PowGate(_g(self), _g(other))

""" below this line, derived gates and operators """

Gate.__pos__ = lambda self: _g(self)
Gate.__neg__ = lambda self: _g(self) * _g(-1)
Gate.__sub__ = lambda self, other: _g(self) + _g(other) * _g(-1)
Gate.__div__ = lambda self, other: _g(self) * (_g(other) ** _g(-1))
Gate.__truediv__ = Gate.__div__


def sqrt(u0):
    return PowGate(_g(u0), _g(-1))
