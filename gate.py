from collections import OrderedDict


class Gate(object):
    """
    Abstracts a differentiable operator in a neural network.
    A gate has a variable number of inputs and knows how to compute its value
    based on them (forward pass), and propagate gradients (backward pass).
    """

    def __init__(self, mnemonic, input_gates):
        self.name = mnemonic
        self.igs = input_gates
        self.val = self.grad = 0.0

    def __repr__(self):
        return '{}[{:.5}, {:.5}]'.format(self.name, self.val, self.grad)

    def forward(self):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

    def dependencies(self):
        """
        Recursively build an evaluation schedule for forward/backward passes.
        Avoid cycles or this will loop forever. The graph network should be partially ordered and connected.
        :return: ordered list of gates, items have no dependency on those preceding them in the list
        """
        dependencies = [g for gate in self.igs for g in gate.dependencies()] + self.igs
        return list(OrderedDict.fromkeys(dependencies))  # dedupe, preserve partial ordering

    def compute(self):
        """
        Recursively computes the gate value (i.e. also updates all gates the current depends on).
        :return: the gate output value
        """
        for gate in self.dependencies() + [self]:
            gate.forward()
        return self.val

    def backprop(self, lr=0.0, grad=1.0):
        """
        Recursively propagates the gradient throughout the gate dependencies.
        By default, it does not update floating network parameters. Use a learning rate > 0 to apply gradients to them.
        Don't use this on hidden (intermediate) nodes (why? because gradient contributions coming from gates
        we do not depend on will be otherwise lost).
        """
        self.grad = float(grad)
        dependencies = self.dependencies()
        for gate in dependencies:
            gate.grad = 0  # reset accumulators for gradients
        for gate in reversed(dependencies + [self]):
            gate.backward()
        self.update_parameters(lr)

    def parameters(self):
        return [gate for gate in self.dependencies() if not gate.igs and gate.floating]

    def update_parameters(self, lr):
        if not lr == 0:
            for param in self.parameters():
                param.val += lr * param.grad

    def checkpoint(self):
        return [gate.val for gate in self.parameters()]

    def restore_checkpoint(self, checkpoint):
        for gate, val in zip(self.parameters(), checkpoint):
            gate.val = val
