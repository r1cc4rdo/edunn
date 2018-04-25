"""
Not fully forned ideas here.
"""

from random import random
from iogates import Constant, Gate


class EndCap(Gate):

    def __init__(self, nodes):
        super(EndCap, self).__init__(nodes)

    def forward(self):
        self.val = [gate.value for gate in self.igs]

    def backward(self):
        for gate in self.igs:
            gate.grad += self.grad


class TrainingExample(object):

    def __init__(self):
        self.a = Constant(0)
        self.b = Constant(0)

    def next_example(self):
        self.a.val = random()
        self.b.val = random()

    def input_gates(self):
        return [self.a, self.b]
