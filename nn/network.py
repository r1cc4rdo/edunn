from random import random
from collections import OrderedDict

from gates import *

# TODO: sugar: call with bind
# TODO: init and update as classes


class Net(object):
    """
    A network object knows how to perform forward and backward passes on a gate graph.
    This is the pseudo-code for a typical usage scenario:

        n = Net(root_gate)
        n.init_parameters(...)
        while training
            n.compute(...)  # forward pass
            n.reset_gradients(...)
            n.backprop(...)  # backward pass
            n.update_parameters(...)

    See linear_classifier_constraint.py for a canonical example.
    """
    def __init__(self, graph):
        """

        """
        def recurse(current_gate):
            return [g for input_gate in current_gate.igs for g in recurse(input_gate)] + [current_gate]
        self.gates = list(OrderedDict.fromkeys(recurse(graph)))  # dedupe, preserve partial ordering
        self.parameters = {g.alias: g for g in self.gates if isinstance(g, Parameter)}
        self.inputs = {g.alias: g for g in self.gates if isinstance(g, Input)}

    def init_parameters(self, values_dict=None):
        for alias in values_dict:
            self.parameters[alias].val = values_dict[alias] if values_dict else random() - 0.5

    def set_inputs(self, inputs_dict=None):
        for alias in inputs_dict:
            self.inputs[alias].val = inputs_dict[alias]

    def compute(self):
        for g in self.gates:
            g.forward()
        return self.gates[-1].val

    def reset_gradients(self, do_not_reset_parameters=False):
        for g in self.gates:
            skip = g in self.parameters and do_not_reset_parameters
            g.grad = g.grad if skip else np.zeros_like(g.val)

    def backprop(self, grad):
        self.gates[-1].grad = grad
        for g in reversed(self.gates):
            g.backward()

    def update_parameters(self, lr):
        for g in list(self.parameters.values()):
            g.val += lr * g.grad
