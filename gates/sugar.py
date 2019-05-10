"""
Syntactic sugar for using gates.
"""
import gates


Gate.__add__ = lambda self, other: Add(_g(self), _g(other))
Gate.__mul__ = lambda self, other: Mul(_g(self), _g(other))
Gate.__pow__ = lambda self, other: Pow(_g(self), _g(other))
Gate.__rpow__ = lambda self, other: Pow(_g(other), _g(self))

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
