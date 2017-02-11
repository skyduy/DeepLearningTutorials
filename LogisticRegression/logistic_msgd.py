#!/usr/bin/python
# coding: utf-8

"""
    Minibatch Stochastic gradient descent
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""
from __future__ import print_function

import cPickle
import os
import sys
import timeit

import theano
import numpy as np
import theano.tensor as T
from utils import load_data
rng = np.random


class LogisticRegression(object):
    def __init__(self, x_in, n_in, n_out):
        """
        :param x_in: 训练数据
        :param n_in: 特征个数
        :param n_out: 类别个数
        """
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        # （样本数，特征数）点乘（特征数，类别数） + （1，类别数）
        self.p_y_given_x = T.nnet.softmax(T.dot(x_in, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = x_in

    def negative_log_likelihood(self, y):
        """
        :return: softmax代价
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def sgd_optimization_mnist(learning_rate=0.1126, n_epochs=1000, batch_size=600, dataset='mnist.pkl.gz'):
    """
    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type batch_size: int
    :param batch_size: 随机梯度下降batch块大小

    :type dataset: string
    :param dataset: the path of the MNIST dataset file

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # build model
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    classifier = LogisticRegression(x_in=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W), (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost,
    # and in the same time updates the parameter of the model based on the rules defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }  # 在变量cost中会接收y（定义cost时用到的classifier接收x）故这里提供x和y值
    )

    # 测试模型
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # 验证模型
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # train model
    print('... training the model')
    # early-stopping，防止过拟合
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)  # 交叉验证频率（迭代多少次minibatch验证一次）

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while epoch < n_epochs and not done_looping:
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            train_model(minibatch_index)
            iter_num = (epoch - 1) * n_train_batches + minibatch_index  # 以minibatch为单位，迭代次数

            if (iter_num + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation acc %.2f %%' %
                    (epoch, minibatch_index + 1, n_train_batches, (1-this_validation_loss) * 100.)
                )

                # 检查交叉验证结果是否更好
                if this_validation_loss < best_validation_loss:
                    # 若足够好，便提高结束次数上限
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)
                    best_validation_loss = this_validation_loss

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        '     epoch %i, minibatch %i/%i, test acc of best model %.2f %%' %
                        (epoch, minibatch_index + 1, n_train_batches, (1-test_score) * 100.)
                    )

                    with open('model/best_msgd_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter_num:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        'Optimization complete with best validation acc of %.2f %%, with test performance %.2f %%' %
        ((1-best_validation_loss) * 100., (1-test_score) * 100.)
    )

    print(
        'The code run for %d epochs, with %.2f epochs/sec' %
        (epoch, 1. * epoch / (end_time - start_time))
    )

    print(
        'The code for file %s ran for %.1fs' % (os.path.split(__file__)[1], end_time - start_time),
        file=sys.stderr
    )


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
    # load the saved model
    classifier = cPickle.load(open('model/best_msgd_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
    print("True answer:")
    print(test_set_y[:10].eval())

if __name__ == '__main__':
    sgd_optimization_mnist()
    predict()
