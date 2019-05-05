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

from gates.gate import Gate


class Leaf(Gate):
    """
    Superclass for all leaf nodes.
    No input gates, so nothing to do in forward and backward passes.
    """
    def __init__(self, name):
        super(Leaf, self).__init__(name, [])

    def forward(self):
        pass

    def backward(self):
        pass


class NamedLeaf(Leaf):
    """
    Superclass to input and parameter variables. The difference between input and parameter is only semantic.
    For example, in f(x, y) = relu(a * x + b * y + c) the variables x, y are an input tuple.
    They will have meaningful gradients, but we are not interested in updating their value.
    a, b, c on the contrary are parameters (weights) to the model, optimized during training.
    """
    def __init__(self, name, alias):
        super(NamedLeaf, self).__init__(name)
        self.alias = alias

    def __repr__(self):
        return '{}({})_{}/{}'.format(self.name, self.alias, self.val, self.grad)

    def __str__(self):
        return self.alias if self.alias else self.name


class Constant(Leaf):

    def __init__(self, value):
        super(Constant, self).__init__('const')
        self.val = value


class Input(NamedLeaf):
    """
    A variable with a name, meant as a model input (e.g. training example).
    See Named for a more in-depth discussion of the differences with Parameter.
    """
    def __init__(self, alias):
        super(Input, self).__init__('input', alias)


class Parameter(NamedLeaf):
    """
    A variable with a name, meant as a model parameter (e.g. weight).
    See Named for a more in-depth discussion of the differences with Input.
    """
    def __init__(self, alias=''):
        super(Parameter, self).__init__('param', alias)


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
