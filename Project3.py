import os
import warnings
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data

# pre-config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
tfe.enable_eager_execution()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

dim_hidden = 1028

# Layer from prof DONOT CHANGE
layer_pool4x4 = tf.layers.MaxPooling2D(4, 4)

layer_cnn0 = tf.layers.Conv2D(
    filters=8,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    # strides=[2,2]
)
layer_pool1x1 = tf.layers.MaxPooling2D(pool_size=[1, 1], strides=1)
layer_pool2x2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
layer_pool3x3 = tf.layers.MaxPooling2D(pool_size=[3, 3], strides=3)
layer_cnn1 = tf.layers.Conv2D(
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    # strides=[2,2]
)
layer_cnn2 = tf.layers.Conv2D(
    filters=1024,
    kernel_size=[2, 2],
    padding="same",
    activation=tf.nn.relu,
    # strides=[2,2]
)
layer_flatten = tf.layers.Flatten()
layer_fc0 = tf.layers.Dense(dim_hidden, activation=tf.nn.relu)
layer_dropout = tf.layers.Dropout(rate=.75)  # dropout rate is 0.75. Retain 0.25
layer_fc1 = tf.layers.Dense(10, activation=None)  # 1


# forward propagation
def prediction(X, training):
    # Instantiate
    values = tf.constant(X)
    # First Layer from Prof
    values = layer_pool4x4(values)
    # Conv layer 1
    values = layer_cnn0(values)
    # Conv layer 2
    values = layer_cnn1(values)
    # Conv layer 3
    values = layer_cnn2(values)
    # Fully connected
    values = layer_flatten(values)
    values = layer_fc0(values)
    values = layer_dropout(values, training=training)
    values = layer_fc1(values)
    return values


# cross entropy loss
def loss(X, y, training):
    logits = prediction(X, training)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)
    return loss


def binary_accuracy(X, y):
    logits = prediction(X, training=False)
    predict = tf.argmax(logits, 1).numpy()
    target = np.argmax(y, 1)
    binary_accuracy = np.sum(predict == target) / len(target)
    return (binary_accuracy)


X_validation = mnist.validation.images
y_validation = mnist.validation.labels
X_validation = X_validation.reshape([-1, 28, 28, 1])


def v_binary_accuracy():
    return (binary_accuracy(X_validation, y_validation))

batch_size = 200
iters = 500

for i in range(iters):
    X, y = mnist.train.next_batch(batch_size)
    X = X.reshape([-1, 28, 28, 1])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    optimizer.minimize(lambda: loss(X, y, True))

    if i % 100 == 0:
        batch_accuracy = binary_accuracy(X, y)
        validation_accuracy = v_binary_accuracy()
        print("batch %d, batch accuracy %.3f validation accuracy %.3f" %
              (i, batch_accuracy, validation_accuracy))

# evaluate the result
X, y = mnist.test.images, mnist.test.labels
X = X.reshape([-1, 28, 28, 1])
test_accuracy = binary_accuracy(X, y)
print("test accuracy %g" % (test_accuracy))