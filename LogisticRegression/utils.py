#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import os
import gzip
import theano
import cPickle
import numpy as np
import theano.tensor as T


def load_data(fp):
    """
    Loads the dataset

    :type fp: string
    :param fp: the path to the dataset (here MNIST)
    """

    data_dir, data_file = os.path.split(fp)
    # 若调用该函数的文件目录下没有该文件，则默认从上层目录的data目录下寻找
    if data_dir == "" and not os.path.isfile(fp):
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", fp)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            fp = new_path
    # 若加载的是mnist且不存在，则下载并保存
    if (not os.path.isfile(fp)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, fp)

    print '... loading data'

    f = gzip.open(fp, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix) which row's correspond to an example.
    # target is a numpy.ndarray of 1 dimensions (vector)) that have the same length as the number of rows in the input.
    # It should give the target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """
            Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # shared_y作为label时应该为int类型
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval
