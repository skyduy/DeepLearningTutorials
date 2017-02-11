#!/usr/bin/python
# coding: utf-8

"""
    Accuracy use activation tanh:
        98.2% after 500 epoch. (Manual stop. It takes too much time...)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""
from __future__ import print_function
import os
import sys
import timeit

import numpy
import theano
import theano.tensor as T
from LogisticRegression.utils import load_data
from LogisticRegression.logistic_msgd import LogisticRegression


class HiddenLayer(object):
    def __init__(self, rng, x_in, n_in, n_out, activation=T.tanh):
        """
        :param rng: a random number generator used to initialize weights
        :param x_in: input data: （样本数，特征数）
        :param n_in: 特征数
        :param n_out: 隐含层特征数
        :param activation: 激活函数
        """
        self.input = x_in

        # 参数初始化
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),  # 防止对称失效
            dtype=theano.config.floatX
        )
        if activation == T.nnet.sigmoid:
            # 一定程度上防止梯度消失 因为sigmoid的导数最大值为1/4
            W_values *= 4
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # 激活选择
        lin_output = T.dot(x_in, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, rng, x_in, n_in, n_hidden, n_out, activation=T.tanh):
        self.hiddenLayer = HiddenLayer(rng, x_in, n_in, n_hidden, activation)

        # The logistic regression layer gets the hidden units of the hidden layer as input
        self.logRegressionLayer = LogisticRegression(
            x_in=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm and square of L2 norm
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = x_in

        # 预测值
        self.y_pred = self.logRegressionLayer.y_pred


def mlp_train(learning_rate=0.01, n_hidden=500, batch_size=20, n_epochs=1000,
             L1_reg=0.00, L2_reg=0.0001, activation=T.tanh, dataset='mnist.pkl.gz'):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # BUILD MODEL
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, x_in=x, n_in=28 * 28, n_hidden=n_hidden, n_out=10, activation=activation)

    # loss function value
    cost = (
        classifier.negative_log_likelihood(y) +
        L1_reg * classifier.L1 +
        L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of (variable, update expression) pairs
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost,
    # and in the same time updates the parameter of the model based on the rules defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TRAIN MODEL
    print('... training')

    # early-stopping 解释见 LogisticRegression.logistic_msgd
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while epoch < n_epochs and not done_looping:
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            train_model(minibatch_index)
            # iteration number
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation acc %f %%' %
                    (epoch, minibatch_index + 1, n_train_batches, (1-this_validation_loss) * 100.)
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter_num

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        '     epoch %i, minibatch %i/%i, test acc of best model %f %%' %
                        (epoch, minibatch_index + 1, n_train_batches, (1-test_score) * 100.)
                    )

            if patience <= iter_num:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        'Optimization complete. Best validation score of %f %% obtained at iteration %i, with test performance %f %%' %
        (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print(
        'The code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_time - start_time)/60.),
        file=sys.stderr
    )


if __name__ == '__main__':
    act1 = T.tanh
    act2 = T.nnet.relu  # not test yet
    act3 = T.nnet.sigmoid  # not test yet
    mlp_train(activation=act1)
