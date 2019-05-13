from functools import reduce

import numpy as np

from gates.add import Arity2Plus


class Mul(Arity2Plus):
    """
    >>> from gates import *
    >>> a, b = (Const(x) for x in (-2, range(3)))
    >>> aa, ab, ba, bb =  (Mul(x, y) for x in (a, b) for y in (a, b))
    >>> tuple(v for g in (aa, ab, ba, bb) for _, v in ((g.forward(), g.val), ))
    (4, array([ 0, -2, -4]), array([ 0, -2, -4]), array([0, 1, 4]))

    >>> ab.grad, a.grad, b.grad = 0.123, 0., np.zeros_like(b.val, np.float)
    >>> ab.backward()
    >>> a.grad
    0.369
    >>> b.grad
    array([ 0.   , -0.246, -0.492])
    """
    name = 'mul'

    def forward(self):
        self.val = reduce(np.multiply, (gate.val for gate in self.igs))

    def backward(self):
        for gate in self.igs:
            prod_wo_gate = reduce(np.multiply, (g.val for g in self.igs if g is not gate))
            for idx_input_grad, idx_pwg, idx_self_grad in Mul.broadcast_indexes(gate.val, prod_wo_gate):
                gate.grad[idx_input_grad] += prod_wo_gate[idx_pwg] * self.grad[idx_self_grad]


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
