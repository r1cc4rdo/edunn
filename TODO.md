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