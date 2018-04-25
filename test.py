from sugar import *

a, b, c, d = param([1.0, 2.0, 3.0, 4.0], lr=0.01)
x = (a + b) / (c + d)

assert(x.check_numerical_gradient(verbose=True))
print '---'
print 'Initial output: {:}'.format(x.compute())
x.backprop()
print 'Final output: {:.5}'.format(x.compute())
