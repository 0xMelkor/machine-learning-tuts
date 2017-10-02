"""
      _             _                              _                   
     | |           | |                            | |                  
  ___| |_ __ _  ___| | __  ___ _ __ ___   __ _ ___| |__   ___ _ __ ___ 
 / __| __/ _` |/ __| |/ / / __| '_ ` _ \ / _` / __| '_ \ / _ \ '__/ __|
 \__ \ || (_| | (__|   <  \__ \ | | | | | (_| \__ \ | | |  __/ |  \__ \
 |___/\__\__,_|\___|_|\_\ |___/_| |_| |_|\__,_|___/_| |_|\___|_|  |___/

@author: Andrea Simeoni 02 ott 2017   
https://github.com/insanediv/machine-learning-tuts/blob/master/basic/softmax_regression.py
"""
import os
import os.path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

del os.environ['TCL_LIBRARY']


def load_mnist_dataset():
    """
    :return: MNIST dataset loaded from default directory 
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mnist_data_path = os.path.join(dir_path, 'mnist_data')
    return input_data.read_data_sets(mnist_data_path, one_hot=True)


# Parameters
learning_rate = 0.5
num_epochs = 1000
batch_size = 100

# Import data
mnist = load_mnist_dataset()

# Define the model
x = tf.placeholder(tf.float32, [None, 784])  # Input images
y_ = tf.placeholder(tf.float32, [None, 10])  # Input labels

# Variables to be trained
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Evidence
y = tf.matmul(x, W) + b

# Define loss and optimizer

# Use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# Train the model
with tf.Session() as session:
    session.run(init)
    losses = []
    for epoch in range(num_epochs):
        # Get next batch
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Get current loss for plotting
        _l = session.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
        losses.append(_l)
        print("Current loss: %s" % _l)
        # Train to get better loss
        session.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # Correctness test (prevision == label)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # How many of the made predictions are correct ?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_value = session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print(accuracy_value)

    # Plot loss
    plt.figure(1)
    plt.subplot(211)
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig('softmax_loss.png')

    plt.close('all')

    # Plot weights
    # Get weights for each class
    class_weights = []
    for class_index in range(10):
        sliced = session.run(tf.slice(session.run(W), [0, class_index], [784, 1]))
        weights = session.run(tf.squeeze(sliced))
        class_weights.append(weights.reshape([28, 28]).tolist())

    fig, graph_list = plt.subplots(nrows=2, ncols=5)

    flatten_graph_list = []
    for sublist in graph_list:
        for item in sublist:
            flatten_graph_list.append(item)

    for graph_index in range(len(flatten_graph_list)):
        graph = flatten_graph_list[graph_index]
        graph.imshow(class_weights[graph_index])
        graph.axis('off')

    fig.suptitle('Learned weights per class', fontsize=20)
    plt.savefig('softmax_weights.png')
