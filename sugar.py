"""
Syntactic sugar for using gates.
"""

from iogates import *
from opgates import *
from neuron import NeuronGate
from collections import Iterable


def const(values):
    if not isinstance(values, Iterable):
        return Constant(values)
    return (Constant(v) for v in values)


def param(values, lr):
    if not isinstance(values, Iterable):
        return Parameter(values, lr)
    return (Parameter(v, lr) for v in values)


def mul(u0, u1):
    return MulGate(u0, u1)


def add(u0, u1):
    return AddGate(u0, u1)


def div(u0, u1):
    return DivGate(u0, u1)


def relu(u0):
    return ReluGate(u0)


def sigmoid(u0):
    return SigmoidGate(u0)


def neuron(a, b, c, x, y):
    return NeuronGate(a, b, c, x, y)


Gate.__add__ = lambda self, other: add(self, other)
Gate.__mul__ = lambda self, other: mul(self, other)
Gate.__div__ = lambda self, other: div(self, other)
Gate.__truediv__ = Gate.__div__
