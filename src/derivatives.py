import numpy as np
import theano
import theano.tensor as T
from theano import pp
x = T.dscalar('x')
y = np.sin(x)/(np.exp(x)+np.cos(x))
gy = T.grad(y, x)
f = theano.function([x], gy)
print(f(6))
print(np.cos(6))


x = T.dvector('x')
y = x ** 2
J, updates = theano.scan(lambda i, y, x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
f = theano.function([x], J, updates=updates)
print(f([4, 4]))