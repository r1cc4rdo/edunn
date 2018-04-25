import math


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Check if two values are equal up to a relative and/or absolute precision.
    From Python 3.5 math module
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def check_gradients(gate, eps=1e-09, verbose=False):
    """
    For each parameter of the network, verify if numerical and analytical gradients are matching.
    :return: True if the test passes, False otherwise
    """
    out = gate.compute()  # populate values
    gate.backprop()  # populate gradients, don't updates parameters

    for param in gate.parameters():

        prev_val, param.val = param.val, param.val + eps
        new_output = gate.compute()
        der = (new_output - out) / eps

        if verbose:
            print 'Param value: {:.5}, Analytical grad: {:.5}, Numerical grad: {:.5}, Diff: {:.5}'.format(
                prev_val, float(param.grad), der, float(param.grad) - der)

        param.val = prev_val  # restore original value
        if not isclose(param.grad, der, rel_tol=math.sqrt(eps)):
            return False

    gate.compute()  # restore original values for all gates
    return True
