#!/usr/bin/env python
# coding=utf8

import tensorflow as tf


def main():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)

    sess = tf.Session()
    result = sess.run(product)
    print result
    sess.close()  # 执行该句以释放资源

    # 如下方式执行,执行后会自动释放资源
    with tf.Session() as sess:
        result = sess.run(product)
        print result


if __name__ == '__main__':
    main()
