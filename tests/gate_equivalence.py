import numpy as np
from utils.sugar import *
from numerical_gradients import randomize_inputs


def test_equivalent_pair(ga, gb, verbose=False, random_trials_per_gate=100):
    """
    Verifies that gates A and B compute the same function and produce the same gradients.
    They are assumed to the hooked to the same set of input gates (i.e. set(ga.parameters()) == set(gb.parameters()))
    """
    for _ in range(random_trials_per_gate):

        randomize_inputs(ga)  # shares inputs with gb
        try:

            vala = ga.compute()
            ga.backprop(grad=0.42)
            grada = [g.grad for g in ga.parameters()]

            valb = gb.compute()
            gb.backprop(grad=0.42)
            gradb = [g.grad for g in gb.parameters()]

            assert(np.allclose(vala, valb))
            assert(np.allclose(grada, gradb))

        except Exception as e:
            print(e, ga, gb, ga.parameters())
            return False

    return True


def test_all_equivalent_gates(verbose=False):
    """
    Verifies forward and backward passes by checking equivalent gate chains that perform the same computation.
    E.g. $norm(a, b, c)$ should always be equal in value and gradients to $sqrt(a*a + b*b + c*c)$.
    """
    a, x, b, y, c = param(0, 0, 0, 0, 0)
    tests = (test_equivalent_pair(a - b,
                                  a + (-b), verbose),
             test_equivalent_pair(a * 2,
                                  a + a + a - a, verbose),
             test_equivalent_pair(a / 3,
                                  a * (1.0 / 3.0), verbose),
             test_equivalent_pair(minimum(a, b),
                                  -maximum(-a, -b), verbose),
             test_equivalent_pair(norm(a, b, c),
                                  sqrt(a*a + b*b + c*c), verbose),
             test_equivalent_pair(neuron(a, x, b, y, c),
                                  sigmoid(a * x + b * y + c), verbose))
    return len(tests) - sum(tests), len(tests)  # failures, total


if __name__ == '__main__':
    print('{} failures out of {} tests performed'.format(*test_all_equivalent_gates()))
