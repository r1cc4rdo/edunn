import sys
from random import random
from sugar import *


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Check if two values are equal up to a relative and/or absolute precision.
    From Python 3.5 math module
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def check_gradients(gate, verbose=False):
    """
    For each parameter of the network, verify if numerical and analytical gradients are matching.
    :return: True if the test passes, False otherwise
    """
    out = gate.compute()  # populate values
    gate.backprop()  # populate gradients, don't updates parameters

    eps = 1e3 * sys.float_info.epsilon
    for param in gate.parameters():

        prev_val, param.val = param.val, param.val + eps
        new_output = gate.compute()
        der = (new_output - out) / eps

        if verbose:
            print 'Param value: {:.5}, Analytical grad: {:.5}, Numerical grad: {:.5}, Abs. Diff: {:.5}'.format(
                prev_val, float(param.grad), der, float(param.grad) - der)

        param.val = prev_val  # restore original value
        if not isclose(param.grad, der):
            return False

    gate.compute()  # restore original values for all gates
    return True


def test_gates(verbose=False):
    """
    Verifies that numerical and analytical gradient do match for all gate types, on all inputs.
    """
    a, b, c, x, y = param((0, 0, 0, 0, 0))
    gates = [a * b, a + b, a / b, sigmoid(a), minimum(a, b),
             maximum(a, b), relu(a), neuron(a, b, c, x, y)]

    eps = sys.float_info.epsilon
    for gate in gates:

        print '\nTesting {}\n'.format(gate.name)
        for _ in range(10):

            for input_gate in gate.parameters():
                input_gate.val = random() * 20 - 10
                input_gate.val = input_gate.val if random() < 0.95 else eps if random() < 0.5 else -eps

            check_gradients(gate, verbose=verbose)


if __name__ == '__main__':
    test_gates(verbose=True)
