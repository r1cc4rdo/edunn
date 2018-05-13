import sys
import inspect
from random import random

import gates as gm  # gates module
from sugar import *


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Check if two values are equal up to a relative and/or absolute precision.
    From Python 3.5 math module
    """
    couples = zip(a, b) if isinstance(a, Iterable) else ((a, b),)
    return all(abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol) for x, y in couples)


def check_gradients(gate, verbose=False):
    """
    For each parameter of the network, verify if numerical and analytical gradients are matching.
    :return: True if the test passes, False otherwise
    """
    delta = 1e-5
    grad = random() - 0.42  # avoiding grad == 1, which can hide some classes of errors
    output = gate.compute()  # populate values
    for input_param in gate.parameters():

        prev_val, input_param.val = input_param.val, input_param.val + delta
        new_output = gate.compute()
        gate.backprop(grad=grad)  # populate gradients
        der = grad * (new_output - output) / delta

        if verbose:
            print 'Param value: {:.5}, Analytical grad: {:.5}, Numerical grad: {:.5}, Abs. Diff: {:.5}'.format(
                prev_val, float(input_param.grad), der, float(input_param.grad) - der)

        input_param.val = prev_val  # restore original value
        if not isclose(input_param.grad, der, rel_tol=1e-6, abs_tol=1e-5):

            print gate
            print gate.igs

            return False

    gate.compute()  # restore original values for all gates
    return True


def test_gates(verbose=False):
    """
    Verifies that numerical and analytical gradient do match for all gate types, on all inputs.
    """
    random_trials_per_gate = 10
    eps = sys.float_info.epsilon
    params = tuple(param((0, 0, 0, 0, 0)))

    module_contents = [getattr(gm, name) for name in dir(gm)]
    module_gates = [g for g in module_contents if inspect.isclass(g) and issubclass(g, gm.Gate)]
    module_gates = [g for g in module_gates if g not in (gm.Parameter, gm.Constant, gm.Gate)]
    for gate_class in module_gates:

        init_args = inspect.getargspec(gate_class.__init__)
        arity = len(init_args.args) - 1  # minus 1 because self
        params_to_test = [arity] if init_args.varargs is None else [1, 2, 3]
        for num_params in params_to_test:  # arity of the gate. For variable number of arguments, we test 1..3

            print '\nTesting {}, {} parameter{}'.format(gate_class.__name__, num_params, '' if num_params == 1 else 's')
            gate = gate_class(*params[:num_params])
            for _ in range(random_trials_per_gate):

                for input_gate in gate.parameters():
                    input_gate.val = random() * 3
                    input_gate.val = input_gate.val if random() < 0.95 else 0.0
                    input_gate.val = input_gate.val if random() < 0.5 else -input_gate.val
                    if isinstance(gate, PowGate):
                        pass

                try:
                    assert(check_gradients(gate, verbose=verbose))
                except ValueError as e:
                    print e
                    print gate
                    print gate.igs

            # AddGate [7.084017622562811, 6.9515244613868745, 4.373734311015006]

        # a, b, c, d, e = param((0, 0, 0, 0, 0))
        # gates = [
        #     +a,
        #     -a, NegGate(a),
        #     relu(a), ReluGate(a),
        #     sigmoid(a), SigmoidGate(a),
        #     a ** b, PowGate(a, b),
        #     a + b, AddGate(a), AddGate(a, b, c), AddGate(a, b, c, d),
        #     a * b, MulGate(a), MulGate(a, b, c), MulGate(a, b, c, d),
        #     minimum(a), minimum(a, b), minimum(a, b, c), MinMaxGate(min, a, b),
        #     maximum(a), maximum(a, b), maximum(a, b, c), MinMaxGate(max, a, b),
        #     norm(a), norm(a, b), norm(a, b, c), NormGate(a, b),
        #     neuron(a, b, c, d, e),
        #     a - b,
        #     a / b,
        #     sqrt(a * a)]

    # t1 = norm(a, b, c)
    # t2 = sqrt(a*a + b*b + c*c)


if __name__ == '__main__':
    test_gates(verbose=True)
