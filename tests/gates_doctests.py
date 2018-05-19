import doctest
import inspect
import gates


def run_doc_tests(verbose=False):
    """
    Discovers and runs all doc tests inside the gates module.
    """
    total_failures, total_tests = 0, 0
    gates_objects = [getattr(gates, name) for name in dir(gates)]
    gates_modules = [g for g in gates_objects if inspect.ismodule(g) and g.__package__ == 'gates']
    for gm in gates_modules:
        (failure_count, test_count) = doctest.testmod(gm, verbose=verbose)
        total_failures += failure_count
        total_tests += test_count
    return total_failures, total_tests


if __name__ == '__main__':
    print '{} failures out of {} tests performed'.format(*run_doc_tests())
