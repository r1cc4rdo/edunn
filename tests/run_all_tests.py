import gates_doctests
import numerical_gradients
import gate_equivalence

print 'Running doctests...'
fail, tot = gates_doctests.run_doc_tests(verbose=False)
if fail:
    print '{} failures out of {} tests performed'.format(fail, tot)

print 'Running gate equivalence tests...'
fail, tot = gate_equivalence.test_all_equivalent_gates(verbose=False)
if fail:
    print '{} failures out of {} tests performed'.format(fail, tot)

print 'Running numerical gradient tests...'
fail, tot = numerical_gradients.numerical_vs_analytical_gradients(verbose=False)
if fail:
    print '{} failures out of {} tests performed'.format(fail, tot)
