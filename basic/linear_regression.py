"""
      _             _                              _                   
     | |           | |                            | |                  
  ___| |_ __ _  ___| | __  ___ _ __ ___   __ _ ___| |__   ___ _ __ ___ 
 / __| __/ _` |/ __| |/ / / __| '_ ` _ \ / _` / __| '_ \ / _ \ '__/ __|
 \__ \ || (_| | (__|   <  \__ \ | | | | | (_| \__ \ | | |  __/ |  \__ \
 |___/\__\__,_|\___|_|\_\ |___/_| |_| |_|\__,_|___/_| |_|\___|_|  |___/
                                                                       
@author: Andrea Simeoni 29 set 2017                                                         
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

del os.environ['TCL_LIBRARY']


def get_linear_samples():
    x = np.random.uniform(high=10, low=0, size=100)
    y = 3.5 * x - 4 + np.random.normal(loc=0, scale=2, size=100)
    return x, y


x_train, y_train = get_linear_samples()

# Graph input
X = tf.placeholder(dtype=tf.float32, shape=100)
Y = tf.placeholder(dtype=tf.float32, shape=100)

W = tf.Variable(1.0)
c = tf.Variable(1.0)

# Define hypothesis
Ypred = tf.matmul(W, X) + c

# Define loss function (RMSE)
loss = tf.reduce_mean(tf.square(Ypred - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    steps = dict()
    steps['m'] = []
    steps['c'] = []

    losses = []

    for k in range(1000):
        _m = session.run(W)

        _c = session.run(c)
        _l = session.run(loss, feed_dict={X: x_train, Y: y_train})
        session.run(train, feed_dict={X: x_train, Y: y_train})
        steps['m'].append(_m)
        steps['c'].append(_c)
        losses.append(_l)

        print("Current loss: %s" % _l)

# Plot loss
plt.figure(1)
plt.subplot(211)
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.tight_layout()

w= steps['m'][-1]
b=steps['c'][-1]
estimated_y = w * x_train + b

plt.subplot(212)
plt.plot(x_train, estimated_y)
plt.plot(x_train, y_train, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('output.png')

