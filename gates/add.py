from functools import reduce

import numpy as np

from gates.gate import Gate


class Arity2Plus(Gate):
    """

    """
    arity = (2, None)

    def __init__(self, g0, g1, *argv):
        super().__init__([g0, g1] + list(argv))

    @staticmethod
    def broadcast_indexes(x, y):
        """
        Returns a generator for broadcasting indexes.
        For example, given array x and y of shape (1, 3) and (3, 1) returns ((0,0,0), (1,0,1), (2,0,2),
        (0,1,3), (1,1,4), (2,1,5), (0,2,6), (1,2,7), (2,2,8)). This allows for adding together gradients
        of broadcasted operations.
        """
        ax, ay = np.array(x), np.array(y)
        x_idxs, y_idxs = (np.reshape(np.arange(a.size), a.shape) for a in (ax, ay))
        for ib, (ix, iy) in enumerate(np.broadcast(x_idxs, y_idxs)):
            yield(ix, iy, ib)


class Add(Arity2Plus):
    """
    >>> from gates import *
    >>> a, b = (Const(x) for x in (-2, range(3)))
    >>> aa, ab, ba, bb =  (Add(x, y) for x in (a, b) for y in (a, b))
    >>> tuple(v for g in (aa, ab, ba, bb) for _, v in ((g.forward(), g.val), ))
    (-4, array([-2, -1,  0]), array([-2, -1,  0]), array([0, 2, 4]))

    >>> ab.grad, a.grad, b.grad = np.arange(3), 0., np.zeros_like(b.val, np.float)
    >>> ab.backward()
    >>> a.grad
    0.123
    >>> b.grad
    array([0.123, 0.123, 0.123])
    """
    name = 'add'

    def forward(self):
        self.val = reduce(np.add, (gate.val for gate in self.igs))

    def backward(self):
        for gate in self.igs:
            for idx_ig_grad, _, idx_og_grad in self.broadcast_indexes(gate.val, self.val):
                gate.grad[idx_ig_grad] += self.grad[idx_og_grad]


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
