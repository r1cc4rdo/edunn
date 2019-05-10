import inspect
from random import random

from utils.sugar import *
import gates as gm  # gates module
from utils.numerical import check_gradients


def randomize_inputs(gate, magnitude=10, zero_frequency=0.05):

    for input_gate in gate.parameters():
        input_gate.val = random() * magnitude
        input_gate.val = input_gate.val if random() < (1 - zero_frequency) else 0.0
        input_gate.val = input_gate.val if random() < 0.5 else -input_gate.val
        if isinstance(gate, PowGate):
            gate.igs[0].val = 0.5 + abs(gate.igs[0].val)  # avoid (-1)**0.1 and 0**-1


def numerical_vs_analytical_gradients(verbose=False, random_trials_per_gate=10):
    """
    Verifies that numerical and analytical gradient do match for all gate types, on all inputs.
    """
    attempts, failures = 0, 0
    params = tuple(param(0, 0, 0, 0, 0))

    module_contents = [getattr(gm, name) for name in dir(gm)]
    module_gates = [g for g in module_contents if inspect.isclass(g) and issubclass(g, gm.Gate)]
    module_gates = [g for g in module_gates if g not in (gm.Parameter, gm.Constant, gm.Gate)]
    for gate_class in module_gates:

        init_args = inspect.getargspec(gate_class.__init__)
        arity = len(init_args.args) - 1  # minus 1 because self
        params_to_test = [arity] if init_args.varargs is None else [1, 2, 3]
        for num_params in params_to_test:  # arity of the gate. For variable number of arguments, we test 1..3

            if verbose:
                print('Testing {}, {} parameter(s)'.format(gate_class.__name__, num_params))

            gate = gate_class(*params[:num_params])
            for _ in range(random_trials_per_gate):

                randomize_inputs(gate)
                try:
                    assert(check_gradients(gate, verbose=verbose))
                except Exception as e:
                    print(e, gate, gate.igs)
                    failures += 1

                attempts += 1

    return failures, attempts


if __name__ == '__main__':
    print('{} failures out of {} tests performed'.format(*numerical_vs_analytical_gradients()))
