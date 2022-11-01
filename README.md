# Educational neural networks in Python
This code is loosely inspired to [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/)'s excellent [Hacker's guide to Neural Networks](http://karpathy.github.io/neuralnets/).

This implementation is not a one-to-one translation of the original javascript code into Python, but [there](https://github.com/urwithajit9/HG_NeuralNetwork) [are](https://github.com/johnashu/hackers_guide_to_neural_networks) [many](https://github.com/saiashirwad/Hackers-Guide-To-Neural-Networks-Python) [repositories](https://github.com/pannous/karpathy_neuralnets_python) [on](https://github.com/techniquark/Hacker-s-Guide-to-Neural-Networks-in-Python) [Github](https://github.com/Mutinix/hacker-nn/) that closely match it line-by-line. Use those to follow along the blog post.

The purpose of this version is to simplify network definition and automate the computation of forward and backward passes. Both tasks in Karpathy's document are exploded and manual for clarity's sake.

For example, the linear classifier example can be written as:

```python
from random import choice
from sugar import *

dataset = (((1.2, 0.7), +1.0), ((-0.3, 0.5), -1.0), ((-3.0, -1.0), +1.0),
           ((0.1, 1.0), -1.0), ((3.0, 1.1), -1.0), ((2.1, -3.0), +1.0))

a, b, c = param(1, -2, -1)  # parameters to optimize
x, y, label = const(0, 0, 0)  # training inputs
f = minimum(1, label * (a * x + b * y + c))

for iteration in range(35000):
    (x.val, y.val), label.val = choice(dataset)
    f.compute()  # forward pass
    f.backprop(0.1)  # backward pass

print 'a, b, c = {:.2f}, {:.2f}, {:.2f}'.format(a.val, b.val, c.val)
```

## TODOs
* Remove numpy, add arrays without it
* Test against JAX
* Separate methods:
  * update()
  * reset()
  * initialize()
  * getType()
  * setValue()

## Features
* Xavier / He init
* Weight decay
* Minibatch GD

## Examples
* [Karpathy’s Hacker’s guide to Neural Networks](http://karpathy.github.io/neuralnets/)
* [Joel Grus](https://github.com/joelgrus/joelnet) [youtube](https://www.youtube.com/watch?v=o64FV-ez6Gw)
* [Justin Johnson](https://github.com/jcjohnson/pycnn/)
* [victor zhou](https://victorzhou.com/blog/intro-to-cnns-part-2/)
* [James Loy (numpy)](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)
* [Morvan Zhou](https://github.com/MorvanZhou/simple-neural-networks)

## See also
* [geohot/tinygrad](https://github.com/geohot/tinygrad)
* [karpathy/micrograd](https://github.com/karpathy/micrograd)

## Datasets
* [Celsius to Fahrenheit](https://www.thoughtco.com/celcius-to-farenheit-formula-609227)
* [scikit learn datasets](https://scikit-learn.org/stable/datasets.html#datasets)
* [Titanic](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)
* [QMNIST](https://github.com/facebookresearch/qmnist)
