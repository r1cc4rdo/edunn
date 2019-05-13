# TODO

add test with sgd / batch / mini batch gradient descent
record parameters after each iteration for a given seed

1. Accumulate gradients (first / incremental)
2. Run on batch (lists? numpy?)
3. Stopping criteria for backprop
4. RandGate (does it get marginalized? leak weights?)
4. Random input gate
5. numpy instead of values (tests / doctest w/ arrays)
6. input gate with name (compute(name: val))

np.set_printoptions(precision=4, threshold=5, edgeitems=2, sign=' ')

- momentum (nesterov)
- opt. methods
    - stochastic gradient descent
    - mini-batch gradient descent
        - sum
        - avg
    - batch gradient descent
    - rmsprop
    - adam
    - nag
    - adadelta
    - adagrad
- quad. programming
- compile into a callable function?
- animated graph in a notebook
- names for parameters, constants


Restore

    def checkpoint(self):
        return [gate.val for gate in self.parameters()]

    def restore_checkpoint(self, checkpoint):
        for gate, val in zip(self.parameters(), checkpoint):
            gate.val = val

net.input = xxx
Paper notes

## Network class

- net.compute(dict{inputs})
- net.load(file), net.save(file)
- net.cuda(). Numba? Cupy?
- at init, make sure:
    - rename to unique name with suffixes
    - associate unique names with inputs
    - ensure every val is numpy float32 (or whatever, parameterized)
    - ensure gradients have the same shape as vals
    - initialize gradients to zero
    - initialize values of parameters

## Miscellanea

- draw network with ...
- run all tests with discovery
- example learning area of a circle
- use 2to3 to upgrade to python3 https://docs.python.org/2/library/2to3.html
- test gradients with autograd (in test, don't make it a dependency), and with numerical differentiation
- add arity, verify during build, use in test to verify output of array same as collection of invidual gates
- test broadcast for pow (1, 1) (1, n), (n, 1), (n, n).

## CUDA / Acceleration

Beyond Numpy Arrays in Python
https://matthewrocklin.com/blog/work/2018/05/27/beyond-numpy

Basics of CuPy — CuPy 5.4.0 documentation
https://docs-cupy.chainer.org/en/stable/tutorial/basic.html

1.1. A ~5 minute guide to Numba — Numba 0.44.0.dev0+400.g000fbcf-py2.7-linux-x86_64.egg documentation
https://numba.pydata.org/numba-doc/dev/user/5minguide.html

## Visualize graphs

python - How to represent graphs with IPython - Stack Overflow
https://stackoverflow.com/questions/29774105/how-to-represent-graphs-with-ipython

Generating Graph Visualizations with pydot and Graphviz – The Python Haven
https://pythonhaven.wordpress.com/2009/12/09/generating_graphs_with_pydot/

User Guide — graphviz 0.10.1 documentation
https://graphviz.readthedocs.io/en/stable/manual.html

## Scratchpad

docopt/docopt: Pythonic command line arguments parser, that will make you smile
https://github.com/docopt/docopt


"""
In this file we define leaf gates.
A leaf gate has no inputs, and holds a value.
The three types defined here are Constants, Inputs and Parameters.
Only Parameters are affected by gradients, and change throughout training.

>>> i, p, k = Input('in'), Parameter('par'), Constant(3)
>>> i, p, k
(input(in)_nan/nan, param(par)_nan/nan, const_3.0/nan)

>>> import gates.sugar
>>> from nn.network import Net
>>> n = Net(i * p + k)
>>> i.val, p.val = -6, 0.5
>>> n.compute()
array(0.)

>>> n.reset_gradients()
>>> n.backprop(1)
>>> n.update_parameters(0.1)
>>> i, p, k
(input(in)_-6.0/0.5, param(par)_0.5/-6.0, const_3.0/1.0)

sugar aliases

x = range(...)
"""

if __name__ == '__main__':

    # import doctest
    # doctest.testmod(verbose=True)

    import gates.sugar
    from nn.network import Net

    i, p, k = Input('in'), Parameter('par'), Constant(3)
    i.val, p.val = -6, 0.5

    n = Net(i * p + k)
    n.compute()
    n.reset_gradients()
    n.backprop(1)
    n.update_parameters(0.1)


-----------------


    For example, let's try to learn the formula for the area of a circle.
    Given a radius $r$, the area $A$ is equal to: $$A = pi * r ^ e$$ where the
    exponent $e := 2$.

    # >>> import numpy as np
    >>> import gates
    >>> pi = Const(np.pi)
    >>> r = Input('radius')
    >>> e = Weight()
    >>> A = pi * Pow(r, e)
    >>> for r.val in 10 * (np.random((100,)) - 0.5):
    >>>     e.grad = 2
    >>>     A.forward()
    >>>     A.backward()





import doctest
import unittest

import doctest_simple

suite = unittest.TestSuite()
suite.addTest(doctest.DocTestSuite(doctest_simple))
suite.addTest(doctest.DocFileSuite('doctest_in_help.rst'))

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
The tests from each source are collapsed into a single outcome, instead of being reported individually.

$ python doctest_unittest.py

my_function (doctest_simple)
Doctest: doctest_simple.my_function ... ok
doctest_in_help.rst
Doctest: doctest_in_help.rst ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.003s

OK