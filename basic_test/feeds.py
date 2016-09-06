#!/usr/bin/env python
# coding=utf8

import tensorflow as tf


def main():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.mul(input1, input2)

    with tf.Session() as sess:
        print sess.run([output], feed_dict={input1: [7.], input2: [2.]})


if __name__ == '__main__':
    main()
