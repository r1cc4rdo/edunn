import math
from operator import mul
from gates.gate import Gate


class AddGate(Gate):

    def __init__(self, g0, *argv):
        super(AddGate, self).__init__('+', [g0] + list(argv))

    def forward(self):
        self.val = sum(gate.val for gate in self.igs)

    def backward(self):
        for gate in self.igs:
            gate.grad += self.grad


class MulGate(Gate):

    def __init__(self, g0, *argv):
        super(MulGate, self).__init__('*', [g0] + list(argv))

    def forward(self):
        self.val = reduce(mul, (gate.val for gate in self.igs), 1)

    def backward(self):
        for index, gate in enumerate(self.igs):
            all_other_gates = self.igs[:index] + self.igs[index + 1:]
            gate.grad += reduce(mul, (g.val for g in all_other_gates), 1) * self.grad


class NormGate(Gate):

    def __init__(self, g0, *argv):
        super(NormGate, self).__init__('norm', [g0] + list(argv))

    def forward(self):
        self.val = math.sqrt(sum(gate.val**2 for gate in self.igs))

    def backward(self):
        for gate in self.igs:
            gate.grad += (gate.val / self.val) * self.grad if abs(self.val) > 0.0 else self.grad


class MinGate(Gate):

    def __init__(self, g0, *argv):
        super(MinGate, self).__init__('min', [g0] + list(argv))
        self.order_fun = min

    def forward(self):
        self.val = self.order_fun(gate.val for gate in self.igs)

    def backward(self):
        for gate in self.igs:
            if gate.val == self.val:
                gate.grad += self.grad


class MaxGate(MinGate):

    def __init__(self, g0, *argv):
        super(MaxGate, self).__init__(g0, *argv)
        self.name, self.order_fun = 'max', max


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
