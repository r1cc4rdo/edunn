import math
from gate import Gate


class MulGate(Gate):

    def __init__(self, u0, u1):
        super(MulGate, self).__init__('*', [u0, u1])

    def forward(self):
        self.val = self.igs[0].val * self.igs[1].val

    def backward(self):
        self.igs[0].grad += self.igs[1].val * self.grad
        self.igs[1].grad += self.igs[0].val * self.grad


class AddGate(Gate):

    def __init__(self, u0, u1):
        super(AddGate, self).__init__('+', [u0, u1])

    def forward(self):
        self.val = self.igs[0].val + self.igs[1].val

    def backward(self):
        self.igs[0].grad += self.grad
        self.igs[1].grad += self.grad


class DivGate(Gate):

    def __init__(self, u0, u1):
        super(DivGate, self).__init__('/', [u0, u1])

    def forward(self):
        self.val = self.igs[0].val / self.igs[1].val

    def backward(self):
        self.igs[0].grad += self.grad / self.igs[1].val
        self.igs[1].grad += -self.igs[0].val * (self.igs[1].val**-2) * self.grad


class SigmoidGate(Gate):

    def __init__(self, u0):
        super(SigmoidGate, self).__init__('sig', [u0])

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self):
        self.val = self.sigmoid(self.igs[0].val)

    def backward(self):
        s = self.sigmoid(self.igs[0].val)
        self.igs[0].grad += (s * (1 - s)) * self.grad
