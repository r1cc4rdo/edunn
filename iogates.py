from gate import Gate


class Parameter(Gate):

    def __init__(self, value, floating=True):
        super(Parameter, self).__init__([])
        self.floating = floating
        self.val = value

    def forward(self):
        pass

    def backward(self):
        pass


class Constant(Parameter):

    def __init__(self, value):
        super(Constant, self).__init__(value, floating=False)
