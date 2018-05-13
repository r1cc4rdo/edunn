from gates.gate import Gate


class Parameter(Gate):
    """
    A floating value that is optimized during training (i.e. that changes in response to incoming gradients).
    Examples:

    >>> p = Parameter(42)
    >>> p
    par[42.0, 0.0]

    >>> p.compute()
    42.0

    >>> p.val = 3
    >>> p
    par[3.0, 0.0]

    >>> p.backprop(grad=1)
    >>> p
    par[3.0, 1.0]

    >>> p.update_parameters(lr=0.1)
    >>> p
    par[3.1, 1.0]

    >>> p.val = 3
    >>> p.backprop(lr=0.1, grad=-0.123)
    >>> p
    par[2.9877, -0.123]
    """
    def __init__(self, value=0, floating=True):
        super(Parameter, self).__init__('par' if floating else 'const', [])
        self.val, self.floating = value, floating

    def forward(self):
        pass

    def backward(self):
        pass


class Constant(Parameter):
    """
    A value unchanged by incoming gradients.
    Constant does not mean immutable: for example, a Constant gate representing training data will get assigned
    different values between iterations, but the network is not allowed to modify the gate value to improve the
    objective function.
    Examples:

    >>> c = Constant(42)
    >>> c
    const[42.0, 0.0]

    >>> c.compute()
    42.0

    >>> c.val = 3
    >>> c
    const[3.0, 0.0]

    >>> c.backprop(grad=1)
    >>> c
    const[3.0, 1.0]

    >>> c.update_parameters(lr=0.1)
    >>> c
    const[3.0, 1.0]

    >>> c.backprop(lr=0.1, grad=-0.123)
    >>> c
    const[3.0, -0.123]
    """
    def __init__(self, value=0):
        super(Constant, self).__init__(value, floating=False)


if __name__ == '__main__':

    import doctest
    doctest.testmod(verbose=True)
