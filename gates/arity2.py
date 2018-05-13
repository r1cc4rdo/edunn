import math
from gates.gate import Gate


class PowGate(Gate):

    def __init__(self, g0, g1):
        super(PowGate, self).__init__('pow', [g0, g1])

    def forward(self):
        self.val = math.pow(self.igs[0].val, self.igs[1].val)

    def backward(self):
        base, exp = self.igs
        base.grad += exp.val * math.pow(base.val, exp.val - 1) * self.grad
        exp.grad += math.log(base.val) * self.val * self.grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
