#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from LogisticRegression.utils import load_data
from MultilayerPerceptron.mlp import HiddenLayer
from LogisticRegression.logistic_msgd import LogisticRegression


class LeNetConvPoolLayer(object):
    def __init__(self, rng, x_in, image_shape, filter_shape, poolsize=(2, 2)):
        """
        :param rng:  用于初始化权重W的随机种子
        :param x_in:  symbolic image tensor, of shape image_shape
        :param image_shape:  (batch size, num input feature maps, image height, image width)
        :param filter_shape:  (number of filters, num input feature maps, filter height, filter width)
        :param poolsize:  the downsampling (pooling) factor (#rows, #cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = x_in
        # 为tanh激活函数随机初始化W，防止symmetry breaking
        fan_in = np.prod(filter_shape[1:])  # 每个数据的输入节点的个数
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))  # 每个数据的输出节点个数
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # build symbolic expression that computes the convolution of input with filters in W
        conv_out = conv.conv2d(
            input=x_in,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = x_in


def evaluate_lenet5(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=None, batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :param learning_rate: learning rate used (factor for the stochastic gradient)
    :param n_epochs: maximal number of epochs to run the optimizer
    :param dataset: path to the dataset used for training /testing (MNIST here)
    :param nkerns: number of kernels on each layer in list of ints
    :param batch_size: number of batch size
    """
    if nkerns is None:
        nkerns = [20, 50]
    rng = np.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28) to a 4D tensor,
    # compatible with our LeNetConvPoolLayer (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        x_in=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer:
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        x_in=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        x_in=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(x_in=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    # 从这里可见，除了pool层的参数未变（maxpool），其它参数都将反向传导而变化。
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by SGD Since this model has many parameters,
    # it would be tedious to manually create an update rule for each model parameter.
    # We thus create the updates list by automatically looping over all (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print '... training'

    # early-stopping 解释见 LogisticRegression.logistic_msgd
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while epoch < n_epochs and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            iter_num = (epoch - 1) * n_train_batches + minibatch_index
            if iter_num % 100 == 0:
                print 'training @ iter = ', iter_num
            train_model(minibatch_index)
            if (iter_num + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter_num
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print('     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
            if patience <= iter_num:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file %s ran for %.2fm' %
                          (os.path.split(__file__)[1], (end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()
