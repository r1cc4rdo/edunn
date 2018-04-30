from gate import Gate


class MinGate(Gate):

    def __init__(self, u0, u1):
        super(MinGate, self).__init__('min', [u0, u1])

    def forward(self):
        self.val = min(self.igs[0].val, self.igs[1].val)

    def backward(self):
        g0, g1 = self.igs
        gate = g0 if g0.val <= g1.val else g1
        gate.grad += self.grad


class MaxGate(Gate):

    def __init__(self, u0, u1):
        super(MaxGate, self).__init__('max', [u0, u1])

    def forward(self):
        self.val = max(self.igs[0].val, self.igs[1].val)

    def backward(self):
        g0, g1 = self.igs
        gate = g0 if g0.val > g1.val else g1
        gate.grad += self.grad


class ReluGate(Gate):

    def __init__(self, u0):
        super(ReluGate, self).__init__('relu', [u0])

    def forward(self):
        self.val = 0.0 if self.igs[0].val <= 0.0 else self.igs[0].val

    def backward(self):
        self.igs[0].grad += 0.0 if self.igs[0].val <= 0.0 else self.grad
