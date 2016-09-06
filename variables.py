#!/usr/bin/env python
# coding=utf8

import tensorflow as tf


def main():
    state = tf.Variable(0, name="counter")

    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        print sess.run(state)

        for _ in range(3):
            sess.run(update)
            print sess.run(state)


if __name__ == '__main__':
    main()
