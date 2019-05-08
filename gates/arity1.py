import numpy as np

from gates.gate import Gate


class Sigmoid(Gate):
    """
    Sigmoid functions are a class of S-shaped, monotonic, saturating nonlinearities.
    This gate implements the logistic function, defined as: $$S(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}$$
    For $x = 0$, $s(x) = \frac{1}{2}$ and $\frac{ds}{dx} = \frac{1}{4}$.

    >>> import gates
    >>> x = Const(0)
    >>> s = Sigmoid(x)
    >>> s.forward()
    >>> s.val
    array( 0.5)

    >>> x.grad = 0  # resets x gradient (nan by default)
    >>> s.grad = 1
    >>> s.backward()
    >>> x.grad
    array( 0.25)

    Let's compute some ground truth with the following matlab snippet and check results:

    samples = -1:4
    s = @(x) 1 ./ (1 + exp(-x)); % sigmoid
    ds = @(x) exp(x) ./ (exp(x) + 1).^2; % ds/dx
    s(samples), ds(samples)

    >>> import numpy as np
    >>> gt = ((0.268941421369995, 0.196611933241482), (0.500000000000000, 0.250000000000000),
    ...       (0.731058578630005, 0.196611933241482), (0.880797077977882, 0.104993585403507),
    ...       (0.952574126822433, 0.0451766597309121), (0.982013790037908, 0.0176627062132911))
    >>> x.val = range(-1, 5)
    >>> x.val
    array(-1.00, 0.00, 1.00, 2.00, 3.00, 4.00)

    >>> s.forward()  # computes s.val
    >>> s.backward()  # updates x.grad
    >>> np.allclose(gt, zip(s.val, x.grad))  # TODO is this a mistake? should I rewrite x.grad
    True
    """
    def __init__(self, input_gate):
        super(Sigmoid, self).__init__('sig', [input_gate])

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self):
        self.val = self.sigmoid(self.igs[0].val)

    def backward(self):
        self.igs[0].grad += (self.val * (1 - self.val)) * self.grad


class Relu(Gate):
    """
    A (Re)ctified (L)inear (U)nit.
    Behaves like $f(x) = x$ if $x \geq 0$, $x = 0$ otherwise.
    Let's compute the value and gradient of relu(x) for x in -2..2:

    >>> x = Const(range(-2, 3))
    >>> r = Relu(x)
    >>> r.forward()
    >>> r.val
    array(-1.00, 0.00, 1.00, 2.00, 3.00, 4.00)

    >>> x.grad = 0
    >>> r.grad = 0.1
    >>> r.backward()
    >>> x.grad

    >>> [map(float, (v, d)) for v, n, d in [(r.compute(), r.backprop(grad=0.1), x.grad) for x.val in range(-2, 3)]]
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.1], [2.0, 0.1]]

    >>> x.val = range(-2, 3)
    >>> _ = r.compute()
    >>> r.backprop(grad=0.1)
    >>> zip(r.val, x.grad)
    [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.1), (2.0, 0.1)]
    """
    def __init__(self, input_gate):
        super(Relu, self).__init__('relu', [input_gate])

    def forward(self):
        self.val = np.maximum(self.igs[0].val, 0.0)

    def backward(self):
        self.igs[0].grad += (self.igs[0].val > 0.0) * self.grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
