from gate import Gate


class Constant(Gate):

    def __init__(self, value):
        super(Constant, self).__init__([])
        self.val = value

    def forward(self):
        pass

    def backward(self):
        pass


class Parameter(Gate):

    def __init__(self, value, lr):
        super(Parameter, self).__init__([])
        self.val = value
        self.lr = lr

    def forward(self):
        pass

    def backward(self):
        self.val += self.lr * self.grad
