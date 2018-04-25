# Educational neural networks in Python

This code is loosely inspired to [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/)'s excellent [Hacker's guide to Neural Networks](http://karpathy.github.io/neuralnets/).

This implementation is not a one-to-one translation of the original javascript code into Python, but [there](https://github.com/urwithajit9/HG_NeuralNetwork) [are](https://github.com/johnashu/hackers_guide_to_neural_networks) [many](https://github.com/saiashirwad/Hackers-Guide-To-Neural-Networks-Python) [repositories](https://github.com/pannous/karpathy_neuralnets_python) [on](https://github.com/techniquark/Hacker-s-Guide-to-Neural-Networks-in-Python) [Github](https://github.com/Mutinix/hacker-nn/) that closely match it line-by-line. Use those to follow along the blog post.

The main purpose of this version is to simplify network definition and automate the computation of forward and backward passes. Both these tasks are exploded and manual (for clarity's sake!) in Karpathy's blog post.

For example, a single neuron can be written as:

```python
a, b, c = param([1.0, 2.0, -3.0])
x, y = const([-1.0, 3.0])

s = sigmoid(a * x + b * y + c)
print s.compute()  # 0.880797077978
```

Compare it with the original implementation:

```javascript
var a = new Unit(1.0, 0.0);
var b = new Unit(2.0, 0.0);
var c = new Unit(-3.0, 0.0);
var x = new Unit(-1.0, 0.0);
var y = new Unit(3.0, 0.0);

// create the gates
var mulg0 = new multiplyGate();
var mulg1 = new multiplyGate();
var addg0 = new addGate();
var addg1 = new addGate();
var sg0 = new sigmoidGate();

// do the forward pass
var forwardNeuron = function() {
  ax = mulg0.forward(a, x); // a*x = -1
  by = mulg1.forward(b, y); // b*y = 6
  axpby = addg0.forward(ax, by); // a*x + b*y = 5
  axpbypc = addg1.forward(axpby, c); // a*x + b*y + c = 2
  s = sg0.forward(axpbypc); // sig(a*x + b*y + c) = 0.8808
};
forwardNeuron();

console.log('circuit output: ' + s.value); // prints 0.8808
```
