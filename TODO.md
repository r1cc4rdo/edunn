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