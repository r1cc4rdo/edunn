from gate import Gate
from arity1 import SigmoidGate


class NeuronGate(Gate):

    def __init__(self, a, b, c, x, y):
        super(NeuronGate, self).__init__('neuron', [a, b, c, x, y])

    def forward(self):
        a, b, c, x, y = (in_node.val for in_node in self.igs)
        self.val = SigmoidGate.sigmoid(a * x + b * y + c)

    def backward(self):
        partial_grad = self.val * (1 - self.val) * self.grad
        a, b, c, x, y = self.igs
        a.grad += partial_grad * x.val
        x.grad += partial_grad * a.val
        b.grad += partial_grad * y.val
        y.grad += partial_grad * b.val
        c.grad += partial_grad


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
