#!/usr/bin/python
# coding: utf-8

"""
    For smaller datasets and simpler models, more sophisticated descent algorithms can be more effective.
    Here use SciPy’s conjugate gradient solver with Theano on the logistic regression task.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

from __future__ import print_function, division

import os
import sys
import timeit
import theano
import cPickle
import numpy as np
import scipy.optimize
import theano.tensor as T
from utils import load_data


class LogisticRegression(object):
    def __init__(self, x_in, n_in, n_out):
        """
        :param x_in: 训练数据
        :param n_in: 特征个数
        :param n_out: 类别个数
        """

        self.theta = theano.shared(
            value=np.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
            ),
            name='theta',
            borrow=True
        )  # Tip: 待update的对象要为shared类型。
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]

        # （样本数，特征数）点乘（特征数，类别数） + （1，类别数）
        self.p_y_given_x = T.nnet.softmax(T.dot(x_in, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

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


def cg_optimization_mnist(n_epochs=50, batch_size=600, mnist_pkl_gz='mnist.pkl.gz'):
    datasets = load_data(mnist_pkl_gz)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    n_in = 28 * 28  # number of input units
    n_out = 10  # number of output units

    # build model
    print('... building the model')

    # allocate symbolic variables for the data
    minibatch_offset = T.lscalar()  # offset to the start of a [mini]batch
    x = T.matrix()   # the data is presented as rasterized images
    y = T.ivector()  # the labels are presented as 1D vector of [int] labels

    # construct the logistic regression class
    classifier = LogisticRegression(x, n_in, n_out)
    cost = classifier.negative_log_likelihood(y)

    # 测试模型
    test_model = theano.function(
        [minibatch_offset],
        classifier.errors(y),
        givens={
            x: test_set_x[minibatch_offset:minibatch_offset + batch_size],
            y: test_set_y[minibatch_offset:minibatch_offset + batch_size]
        },
        name="test"
    )

    # 验证模型
    validate_model = theano.function(
        [minibatch_offset],
        classifier.errors(y),
        givens={
            x: valid_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: valid_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="validate"
    )

    # batch cost模型（objective function 里调用，此处为cost_grad）
    batch_cost = theano.function(
        [minibatch_offset],
        cost,
        givens={
            x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="batch_cost"
    )

    # batch grad模型（objective function 里调用，此处为cost_grad）
    batch_grad = theano.function(
        [minibatch_offset],
        T.grad(cost, classifier.theta),
        givens={
            x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="batch_grad"
    )

    # Objective function.
    def cost_grad(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        train_losses = [batch_cost(i * batch_size) for i in range(n_train_batches)]

        grad = batch_grad(0)
        for i in range(1, n_train_batches):
            grad += batch_grad(i * batch_size)
        return [np.mean(train_losses), grad / n_train_batches]

    validation_scores = [np.inf, 0]

    # callback函数，每次优化改变theta后，均会调用该函数。这里将theta值应用，并更新
    def callback(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        # compute the validation loss
        validation_losses = [validate_model(i * batch_size) for i in range(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
        print(('validation acc %f %%' % ((1-this_validation_loss) * 100.,)))

        # check if it is better then best validation score got until now
        if this_validation_loss < validation_scores[0]:
            # if so, replace the old one, and compute the score on the testing dataset
            validation_scores[0] = this_validation_loss
            test_losses = [test_model(i * batch_size) for i in range(n_test_batches)]
            validation_scores[1] = np.mean(test_losses)
            with open('model/best_cg_model.pkl', 'w') as f:
                cPickle.dump(classifier, f)

    # train model using scipy conjugate gradient optimizer
    print ("Optimizing using scipy.optimize.fmin_cg...")
    start_time = timeit.default_timer()
    opt_solution = scipy.optimize.minimize(
        fun=cost_grad, x0=np.zeros((n_in + 1) * n_out, dtype=x.dtype), method='CG',
        jac=True, callback=callback, options={'maxiter': n_epochs, 'disp': True},
    )  # 这里disp并没有打印什么

    end_time = timeit.default_timer()
    print('Optimization complete with best validation acc of %f %%, with test performance %f %%' %
          ((1-validation_scores[0]) * 100., (1-validation_scores[1]) * 100.))

    print('The code for file %s ran for %.1fs' %
          (os.path.split(__file__)[1], end_time - start_time), file=sys.stderr)


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
    cg_optimization_mnist()
    predict()
