from tensorflow.examples.tutorials.mnist import input_data
data_sets = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import math

x = tf.placeholder(tf.float32, shape=[100, 784], name='x')
y = tf.placeholder(tf.int32, shape=[100, 10], name='y')

with tf.name_scope('h1'):
    w = tf.Variable(tf.truncated_normal([784, 16], stddev=1.0 / math.sqrt(784)), name='w')
    b = tf.Variable(tf.zeros([16]))
    h1 = tf.nn.relu(tf.matmul(x, w) + b)
with tf.name_scope('h2'):
    w = tf.Variable(tf.truncated_normal([16, 16], stddev=1.0 / math.sqrt(16)), name='w')
    b = tf.Variable(tf.zeros([16]))
    h2 = tf.nn.relu(tf.matmul(h1, w) + b)
with tf.name_scope('op'):
    w = tf.Variable(tf.truncated_normal([16, 10], stddev=1.0 / math.sqrt(16)), name='w')
    b = tf.Variable(tf.zeros([10]))
    o = tf.matmul(h2, w) + b

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=y, logits=o, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

global_step = tf.Variable(0, trainable=False)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(o, 1), tf.argmax(y,1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(10000):
        x_labels, y_labels = data_sets.train.next_batch(100)
        _, loss_value = sess.run([train_op, loss], feed_dict={x: x_labels, y: y_labels})
        if i % 100 == 0:
            print('Step %d: loss = %.2f' % (i, loss_value))
            true_count = 0
            steps_per_epoch = 100
            num_examples = 10000
            for step in range(steps_per_epoch):
                x_tests, y_tests = data_sets.train.next_batch(100)
                true_count += sess.run(accuracy, feed_dict={x: x_tests, y: y_tests})
            precision = float(true_count) / num_examples
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                (num_examples, true_count, precision))
