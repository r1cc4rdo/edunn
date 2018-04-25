from collections import OrderedDict
import math

class Gate(object):
    """
    Abstracts a differentiable operator in a neural network.
    A gate has a variable number of inputs and knows how to compute its value
    based on them (forward pass), and propagate gradients (backward pass).
    """

    def __init__(self, input_gates):
        self.igs = input_gates
        self.val = None
        self.grad = 1

    def forward(self):
        """
        Update self.val based on gate inputs.
        """
        raise NotImplementedError()

    def backward(self):
        """
        Propagate self.grad on gate inputs.
        """
        raise NotImplementedError()

    def dependencies(self):
        """
        Recursively build an evaluation schedule for forward/backward passes.
        BEWARE: will loop forever is the graph has cycles! (it shouldn't)
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

    def backprop(self):
        """
        Recursively propagates the gradient throughout the gate dependencies.
        Don't use this on hidden (intermediate) nodes (why? because gradient
        contributions coming from gates we do not depend on will be otherwise lost).
        """
        dependencies = self.dependencies()
        for gate in dependencies:
            gate.grad = 0  # reset accumulators for gradients
        for gate in reversed(dependencies + [self]):
            gate.backward()

    def train(self, passes=1):
        for _ in range(passes):
            self.compute()
            self.backprop()
        self.compute()

    def parameters(self):
        """
        Returns all parameters the gate depends on.
        """
        from iogates import Parameter  # here to avoid circular dependencies
        return [gate for gate in self.dependencies() if isinstance(gate, Parameter)]

    def checkpoint(self):
        """
        Returns the current values of all input parameters as a list.
        """
        return [gate.val for gate in self.parameters()]

    def restore_checkpoint(self, checkpoint):
        """
        Restore a previously recorded checkpoint.
        """
        for gate, val in zip(self.parameters(), checkpoint):
            gate.val = val

    def check_numerical_gradient(self, eps=1e-09, verbose=False):
        """
        For each parameter to the network, verify that numerical and analytical gradients are matching.
        :return: True if the test passes
        """
        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):  # from Python 3.5 math module
            return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        checkpoint = self.checkpoint()
        out = self.compute()  # populate values
        self.backprop()  # populate gradients, but updates parameters
        self.restore_checkpoint(checkpoint)  # restore parameters

        for param in self.parameters():

            prev_val, param.val = param.val, param.val + eps
            new_output = self.compute()
            der = (new_output - out) / eps

            if verbose:
                print 'Param value: {:.5}, Analytical grad: {:.5}, Numerical grad: {:.5}, Diff: {:.5}'.format(
                    prev_val, float(param.grad), der, float(param.grad) - der)

            param.val = prev_val  # restore original value
            if not isclose(param.grad, der, rel_tol=math.sqrt(eps)):
                return False

        self.compute()  # restore original values for all gates
        return True
