"""
Syntactic sugar for using gates.
"""

from iogates import *
from opgates import *
from relgates import *
from neuron import NeuronGate
from collections import Iterable


def const(values):
    if not isinstance(values, Iterable):
        return Constant(values)
    return (Constant(v) for v in values)


def param(values, floating=True):
    if not isinstance(values, Iterable):
        return Parameter(values, floating)
    return (Parameter(v, floating) for v in values)


def _g(gate_or_value):  # upgrade scalar to Constant gate, or return unchanged
    return Constant(gate_or_value) if not isinstance(gate_or_value, Gate) else gate_or_value


def mul(u0, u1):
    return MulGate(_g(u0), _g(u1))


def add(u0, u1):
    return AddGate(_g(u0), _g(u1))


def div(u0, u1):
    return DivGate(_g(u0), _g(u1))


def minimum(u0, u1):  # min is a built-in
    return MinGate(_g(u0), _g(u1))


def maximum(u0, u1):  # max is a built-in
    return MaxGate(_g(u0), _g(u1))


def relu(u0):
    return ReluGate(_g(u0))


def sigmoid(u0):
    return SigmoidGate(_g(u0))


def neuron(a, b, c, x, y):
    return NeuronGate(_g(a), _g(b), _g(c), _g(x), _g(y))


Gate.__add__ = lambda self, other: add(self, other)
Gate.__mul__ = lambda self, other: mul(self, other)
Gate.__div__ = lambda self, other: div(self, other)
Gate.__truediv__ = Gate.__div__
