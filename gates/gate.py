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
        """
        Returns a string identifier for the gate type.
        It's a class property; it does not uniquely identify a single gate or layer in a network.
        :return: string
        """
        pass

    @property
    @abstractmethod
    def arity(self):
        """
        Returns a tuple containing the minimum and maximum number of inputs for the gate (the arity of the operator).
        For example, (1, 1) means the gate takes exactly one input. (2, None) means 2+ inputs.
        :return: (int, int)
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Performs the forward pass, gate.value = function(*gate.igs)
        :return: None
        """
        pass

    @abstractmethod
    def backward(self):
        """
        Perform local back-propagation.
        For each g in gate.igs, g.grad = d/dg function(*gate.igs)
        :return: None
        """
        pass

    def __str__(self):
        """
        Returns the gate class identifier, Gate.name.
        Gate.name is a class property; it does not uniquely identify a single gate or layer in a network.
        :return: string
        """
        return self.name
