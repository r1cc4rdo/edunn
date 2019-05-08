import abc


class Gate(object):
    """
    Abstracts a differentiable operator in a neural network.
    A gate has a variable number of inputs and knows how to compute its value
    based on them (forward pass), and propagate gradients back (backward pass).
    """

    def __init__(self, name, input_gates):
        self.name = name
        self.igs = input_gates
        self.val = self.grad = None

    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def backward(self):
        raise NotImplementedError()

    def __str__(self):
        return self.name
