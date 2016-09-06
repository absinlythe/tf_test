#!/usr/bin/env python
# coding=utf8

import tensorflow as tf


def mnist_regression_evaluate(mnist_data_x, mnist_data_y, sess, x, y, y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print sess.run(accuracy, feed_dict={x: mnist_data_x, y_: mnist_data_y})


def mnist_regression(mnist_data, iter=1000, verbose=True):
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        # init
        sess.run(init)

        # training
        for i in range(iter):
            batch_xs, batch_ys = mnist_data.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if verbose and i % 100 == 0:
                mnist_regression_evaluate(mnist_data.test.images, mnist_data.test.labels, sess, x, y, y_)

        # evaluation
        mnist_regression_evaluate(mnist_data.test.images, mnist_data.test.labels, sess, x, y, y_)


def main():
    from get_mnist_data import get_mnist_data
    import time

    st = time.time()
    mnist_data = get_mnist_data()

    iter = 5000
    mnist_regression(mnist_data, iter=iter, verbose=False)
    et = time.time()
    cost = (et - st)

    print "train finished! cost: %ss, epoch cost: %sms" % (cost, cost / iter * 1000)


if __name__ == '__main__':
    main()
