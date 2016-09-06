#!/usr/bin/env python
# coding=utf8

import tensorflow as tf


def main():
    sess = tf.InteractiveSession()

    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])

    x.initializer.run()

    sub = tf.sub(x, a)
    print sub.eval()

    print sess.run(x)

    sess.close()


if __name__ == '__main__':
    main()
