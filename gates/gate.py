from abc import ABC, abstractmethod


class Gate(ABC):
    """
    Abstracts a differentiable operator in a neural network.
    A gate has a variable number of inputs and knows how to compute its value
    based on them (forward pass), and propagate gradients back (backward pass).
    """

    def __init__(self, input_gates):
        self.igs = input_gates
        self.val = self.grad = None

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __str__(self):
        return self.name
