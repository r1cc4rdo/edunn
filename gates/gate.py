import numpy as np


class Gate(object):
    """
    Abstracts a differentiable operator in a neural network.
    A gate has a variable number of inputs and knows how to compute its value
    based on them (forward pass), and propagate gradients back (backward pass).
    """
    def __init__(self, name, input_gates):
        self.name = name
        self.igs = input_gates
        self.val = self.grad = np.nan

    def forward(self):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        return '{}_{}/{}'.format(self.name, self.val, self.grad)

    def __setattr__(self, attr, value):
        """
        This ensures that the content of value and grad is always a floating type,
        and pushes onto numpy the complexity of handling lists of values.
        """
        value = np.array(value, np.float64) if attr in ('val', 'grad') else value
        super(Gate, self).__setattr__(attr, value)
