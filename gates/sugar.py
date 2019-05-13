"""
Syntactic sugar for using gates.
"""
from gates import *


Gate.__add__ = lambda self, other: Add(self, other)
Gate.__mul__ = lambda self, other: Mul(self, other)
Gate.__pow__ = lambda self, other: Pow(self, other)
Gate.__rpow__ = lambda self, other: Pow(other, self)

Gate.__radd__ = Gate.__add__
Gate.__rmul__ = Gate.__mul__
Gate.__pos__ = lambda self: self
Gate.__neg__ = lambda self: self * -1
Gate.__sub__ = lambda self, other: self + other * -1
Gate.__rsub__ = lambda self, other: other + self * -1
Gate.__div__ = lambda self, other: self * (other ** -1)
Gate.__rdiv__ = lambda self, other: other * (self ** -1)
Gate.__truediv__ = Gate.__div__
Gate.__rtruediv__ = Gate.__rdiv__


def sqrt(u0):
    return PowGate(_g(u0), _g(0.5))
