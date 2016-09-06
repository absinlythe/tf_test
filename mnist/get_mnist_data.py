#!/usr/bin/env python
# coding=utf8

from tensorflow.examples.tutorials.mnist import input_data


def get_mnist_data(train_dir="MNIST_data/"):
    mnist = input_data.read_data_sets(train_dir=train_dir, one_hot=True)
    print "Get mnist data finished. Data dir is %s" % train_dir

    return mnist


def main():
    get_mnist_data()


if __name__ == '__main__':
    main()
