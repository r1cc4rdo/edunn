import numpy as np

from gates.gate import Gate


class Sigmoid(Gate):
    """
    Sigmoid functions are a class of S-shaped, monotonic, saturating nonlinearities.
    This gate implements the logistic function, defined as: $$S(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}$$
    For $x = 0$, $s(x) = \frac{1}{2}$ and $\frac{ds}{dx} = \frac{1}{4}$.

    >>> from gates import *
    >>> x = Const(0)
    >>> s = Sigmoid(x)
    >>> s.forward()
    >>> s.val
    0.5

    >>> x.grad = 0  # resets x gradient (nan by default)
    >>> s.grad = 1
    >>> s.backward()
    >>> x.grad
    0.25

    Let's compute some ground truth with the following matlab snippet and check results:

    samples = -1:4
    s = @(x) 1 ./ (1 + exp(-x)); % sigmoid
    ds = @(x) exp(x) ./ (exp(x) + 1).^2; % ds/dx
    s(samples), ds(samples)

    >>> gt = ((0.268941421369995, 0.196611933241482), (0.500000000000000, 0.250000000000000),
    ...       (0.731058578630005, 0.196611933241482), (0.880797077977882, 0.104993585403507),
    ...       (0.952574126822433, 0.0451766597309121), (0.982013790037908, 0.0176627062132911))
    >>> x.val = np.arange(-1, 5.0)
    >>> x.val
    array([-1.,  0.,  1.,  2.,  3.,  4.])

    >>> x.grad = np.zeros_like(x.val)
    >>> s.forward()  # computes s.val
    >>> s.backward()  # updates x.grad
    >>> np.allclose(gt, zip(s.val, x.grad))
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


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
