from random import random
import numpy as np


def check_gradients(gate, verbose=False):
    """
    For each parameter of the network, verify if numerical and analytical gradients are matching.
    :return: True if the test passes, False otherwise
    """
    delta = 1e-5
    grad = 0.42 if random() > 0.5 else -0.58  # avoiding grad == 1, which can hide some classes of errors
    output = gate.compute()  # populate values
    for input_param in gate.parameters():

        prev_val, input_param.val = input_param.val, input_param.val + delta
        new_output = gate.compute()
        gate.backprop(grad=grad)  # populate gradients
        der = grad * (new_output - output) / delta
        ok = np.allclose(input_param.grad, der, rtol=1e-4, atol=1e-4)

        if verbose or not ok:
            print('Param value: {:.5}, Analytical grad: {:.5}, Numerical grad: {:.5}, Abs. Diff: {:.5}'.format(
                prev_val, float(input_param.grad), der, float(input_param.grad) - der))

        input_param.val = prev_val  # restore original value
        if not ok:
            return False

    gate.compute()  # restore original values for all gates
    return True
