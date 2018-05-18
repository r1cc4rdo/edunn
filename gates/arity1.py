import math
from gates.gate import Gate


class SigmoidGate(Gate):
    """
    Sigmoids are a class of S-shaped, monotonic, saturating non-linearities.
    One example is the logistic function, defined as: $$S(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}$$
    For $x = 0$, $s(x) = \frac{1}{2}$ and $\frac{ds}{dx} = \frac{1}{4}$.

    >>> from utils.sugar import *
    >>> x = param(0)
    >>> s = sigmoid(x)
    >>> s.compute()
    array(0.5)

    >>> s.backprop(grad=1, lr=0.1)
    >>> s, x
    (sig[0.5, 1.0], par[0.025, 0.25])

    Let's compute some ground truth with the following matlab snippet and check results:

    samples = -1:4
    s = @(x) 1 ./ (1 + exp(-x)); % sigmoid
    ds = @(x) exp(x) ./ (exp(x) + 1).^2; % ds/dx
    s(samples), ds(samples)

    >>> import numpy as np
    >>> from operator import add
    >>> gt = ((0.268941421369995, 0.196611933241482), (0.500000000000000, 0.250000000000000),
    ...       (0.731058578630005, 0.196611933241482), (0.880797077977882, 0.104993585403507),
    ...       (0.952574126822433, 0.0451766597309121), (0.982013790037908, 0.0176627062132911))
    >>> sds = [(v, d) for v, n, d in [(s.compute(), s.backprop(), x.grad) for x.val in range(-1, 5)]]
    >>> np.allclose(reduce(add, zip(*gt)), reduce(add, zip(*sds)))
    True
    """
    def __init__(self, g0):
        super(SigmoidGate, self).__init__('sig', [g0])

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self):
        self.val = self.sigmoid(self.igs[0].val)

    def backward(self):
        self.igs[0].grad += (self.val * (1 - self.val)) * self.grad


class ReluGate(Gate):
    """
    A (Re)ctified (L)inear (U)nit.
    Behaves like $f(x) = x$ if $x \geq 0$, $x = 0$ otherwise.
    Let's compute the value and gradient of relu(x) for x in -2..2:

    >>> from utils.sugar import *
    >>> x = param()
    >>> r = relu(x)
    >>> [map(float, (v, d)) for v, n, d in [(r.compute(), r.backprop(grad=0.1), x.grad) for x.val in range(-2, 3)]]
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.1], [2.0, 0.1]]
    """
    def __init__(self, g0):
        super(ReluGate, self).__init__('relu', [g0])

    def forward(self):
        self.val = 0.0 if self.igs[0].val <= 0.0 else self.igs[0].val

    def backward(self):
        self.igs[0].grad += 0.0 if self.igs[0].val <= 0.0 else self.grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
