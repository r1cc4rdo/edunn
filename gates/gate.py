from abc import ABC, abstractmethod

from gates.leaf import Const


class Gate(ABC):
    """
    Abstracts a differentiable operator in a neural network.
    A gate has a variable number of inputs and knows how to compute its value
    based on them (forward pass), and propagate gradients back (backward pass).
    """

    def __init__(self, input_gates):
        self.igs = [ig if isinstance(ig, Gate) else Const(ig) for ig in input_gates]
        self.val = self.grad = None

    @property
    @abstractmethod
    def name(self):
        """
        Returns a string identifier for the gate type.
        This does not uniquely identify each gate/layer in a network.
        :return:
        """
        pass

    @property
    @abstractmethod
    def arity(self):
        """

        :return:
        """
        pass

    @abstractmethod
    def forward(self):
        """

        :return:
        """
        pass

    @abstractmethod
    def backward(self):
        """

        :return:
        """
        pass

    def __str__(self):
        return self.name
