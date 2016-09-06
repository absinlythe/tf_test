#!/usr/bin/env python
# coding=utf8

import tensorflow as tf


def main():
    input1 = tf.constant([3.0])
    input2 = tf.constant([2.0])
    input3 = tf.constant([5.0])
    intermd = tf.add(input2, input3)
    mul = tf.mul(input1, intermd)

    with tf.Session() as sess:
        result = sess.run([mul, intermd])
        print result


if __name__ == '__main__':
    main()
