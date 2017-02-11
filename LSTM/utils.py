#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import numpy as np
import theano
import cPickle
import gzip
import os


def get_dataset_file(dataset, default_dataset, origin):
    """
    Look for it as if it was a full path, if not, try local file, if not try in the data directory.
    Download dataset if it is not present
    """
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        import urllib
        print('Downloading data from %s' % origin)
        urllib.urlretrieve(origin, dataset)
    return dataset


def load_data(fn="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None, sort_by_len=True):
    """Loads the dataset

    :param fn: file name
    :param n_words: 最大词汇量，即单词下标最大值，超出部分设为1，代表unknown.
    :param valid_portion: The proportion of the full train set used for the validation set.
    :param maxlen: 允许最长的句子长度，大于该长度，则忽略不选取
    :param sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    """
    fn = get_dataset_file(fn, "imdb.pkl", "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")
    if fn.endswith(".gz"):
        f = gzip.open(fn, 'rb')
    else:
        f = open(fn, 'rb')
    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # 将数据划分为训练数据和验证数据
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(set_x):
        return [[1 if w >= n_words else w for w in sen] for sen in set_x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    # 将数据集按照句子长度排序
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda index: len(seq[index]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test


def prepare_data(seqs, labels, maxlen=None):
    """
    Create the matrices from the datasets.
    将句子长度统一为maxlen或者最长句子的长度

    :param seqs: 句子序列
    :param labels: label序列
    :param maxlen: 允许句子最大值
    :return: x, x_mask: shape(maxlen, sentence_number)
             labels

    """
    # 统计所有句子的长度
    lengths = [len(s) for s in seqs]

    # 若设置了最大长度，则仅筛选出小于该长度的句子，组成新序列
    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l <= maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    # x_mask用来标记哪些是有效部分
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.

    :param n: 总长度
    :param minibatch_size:
    :param shuffle:
    :return:
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)


def ortho_weight(ndim):
    """
    为什么使用正交初始化weight，参考 https://www.zhihu.com/question/37686246
    """
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)
