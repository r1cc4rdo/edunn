# TODO

1. Accumulate gradients (first / incremental)
2. Run on batch (lists? numpy?)
3. Stopping criteria for backprop
4. RandGate (does it get marginalized? leak weights?)
5. numpy instead of values (tests / doctest w/ arrays)

- momentum (nesterov)
- opt. methods
    - stochastic gradient descent
    - mini-batch gradient descent
        - sum
        - avg
    - batch gradient descent
    - rmsprop
    - adam
    - adadelta
    - adagrad
- quad. programming
- compile into a callable function?
- animated graph in a notebook
- names for parameters, constants