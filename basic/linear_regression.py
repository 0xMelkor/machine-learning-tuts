"""
      _             _                              _                   
     | |           | |                            | |                  
  ___| |_ __ _  ___| | __  ___ _ __ ___   __ _ ___| |__   ___ _ __ ___ 
 / __| __/ _` |/ __| |/ / / __| '_ ` _ \ / _` / __| '_ \ / _ \ '__/ __|
 \__ \ || (_| | (__|   <  \__ \ | | | | | (_| \__ \ | | |  __/ |  \__ \
 |___/\__\__,_|\___|_|\_\ |___/_| |_| |_|\__,_|___/_| |_|\___|_|  |___/
                                                                       
@author: Andrea Simeoni 29 set 2017   
https://github.com/insanediv/machine-learning-tuts/blob/master/basic/linear_regression.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

del os.environ['TCL_LIBRARY']


def gen_linear_samples(n_samples=100):
    x = np.random.uniform(high=10, low=0, size=n_samples)
    y = 3.5 * x - 4 + np.random.normal(loc=0, scale=2, size=n_samples)
    return x, y


# Parameters
num_epochs = 1000
num_samples = 100
learning_rate = 0.01

x_train, y_train = gen_linear_samples(num_samples)

# Graph input
X = tf.placeholder(dtype=tf.float32, shape=num_samples)
Y = tf.placeholder(dtype=tf.float32, shape=num_samples)

# Parameters to be learned
W = tf.Variable(1.0, name="weight")
b = tf.Variable(1.0, name="bias")

# Construct a linear model (Hypothesis)
# h = WX +b
h = tf.add(tf.multiply(X, W), b)

# Define loss function (RMSE)
loss = tf.reduce_mean(tf.square(h - Y))

# Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    steps = dict()
    steps['W'] = []
    steps['b'] = []

    losses = []

    for k in range(num_epochs):
        _W = session.run(W)
        _b = session.run(b)
        _l = session.run(loss, feed_dict={X: x_train, Y: y_train})
        session.run(train, feed_dict={X: x_train, Y: y_train})

        # Store learned params for plotting
        steps['W'].append(_W)
        steps['b'].append(_b)
        losses.append(_l)

        print("Current loss: %s" % _l)

    # Plot loss
    plt.figure(1)
    plt.subplot(211)
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()

    # This is the learned regression line
    learned_w = session.run(W)
    learned_b = session.run(b)
    regression_line = learned_w * x_train + learned_b

    plt.subplot(212)
    plt.plot(x_train, regression_line)
    plt.plot(x_train, y_train, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('output.png')
