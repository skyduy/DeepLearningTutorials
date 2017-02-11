#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""
from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from PIL import Image
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import tile_raster_images
from LogisticRegression.utils import load_data


class DeNosingAutoEncoder(object):
    def __init__(self, np_rng, theano_rng=None, x_in=None, n_visible=784, n_hidden=500, W=None, bhid=None, bvis=None):
        """
        :param np_rng: number random generator used to generate weights
        :param theano_rng: Theano random generator; if None is given one is generated based on a seed drawn from `rng`
        :param x_in: a symbolic description of the input or None for standalone dA
        :param n_visible: number of visible units
        :param n_hidden:  number of hidden units

        :param W: 输入层至隐藏层的权重
        :param bhid: 输入层至隐藏层的偏移量
        :param bvis: 隐藏层至输出层的偏移量
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid  # b corresponds to the bias of the hidden
        self.b_prime = bvis  # b_prime corresponds to the bias of the visible
        self.W_prime = self.W.T  # tied weights, therefore W_prime is W transpose

        self.theano_rng = theano_rng

        # if no input is given, generate a variable representing the input
        if x_in is None:
            # we use a matrix because we expect a minibatch of several examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = x_in

        # 实际参数只有三个，因为使用了tied weights
        # why use tied weights?
        # https://groups.google.com/forum/#!topic/theano-users/QilEmkFvDoE
        # http://stackoverflow.com/questions/36889732/tied-weights-in-autoencoder
        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, x_in, corruption_level):
        """
        降噪自动编码器采用对输入进行随机污染（即引入噪声）的方式来减少学习恒等函数的风险
        This function keeps ``1-corruption_level`` entries of the inputs
        the same and zero-out randomly selected subset of size ``coruption_level``

        Note : first argument of theano.rng.binomial is the shape(size) of random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s
                where 1 has a probability of 1 - ``corruption_level``
                and 0 with ``corruption_level``

                The binomial function return int64 data type by default.
                int64 multiplicated by the input type(floatX) always return float64.
                To keep all data in floatX when floatX is float32, we set the dtype of the binomial to floatX.
                As in our case the value of the binomial is always 0 or 1, this don't change the result.
                This is needed to allow the gpu to work correctly as it only support float32 for now.
        """
        # 这里n为1，即有(1-corruption_level)%的原数据不变，类似dropout见LSTM中的dropout_layer
        return self.theano_rng.binomial(size=x_in.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX) * x_in

    def get_hidden_values(self, x_in):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(x_in, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one training step of the dA """
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return cost, updates


def test_dA(learning_rate=0.1, training_epochs=15, dataset='mnist.pkl.gz', batch_size=20, output_folder='dA_plots'):

    """
    This demo is tested on MNIST
    :param learning_rate: learning rate used for training the DeNosing AutoEncoder
    :param training_epochs: number of epochs used for training
    :param dataset: path to the picked dataset
    :param batch_size: number of batch size
    :param output_folder:
    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    print('build model 1 ...')
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = DeNosingAutoEncoder(np_rng=rng, theano_rng=theano_rng, x_in=x, n_visible=28 * 28, n_hidden=500)
    cost, updates = da.get_cost_updates(corruption_level=0., learning_rate=learning_rate)
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()
    print('training 1...')
    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print('Training epoch %d, cost ' % epoch, np.mean(c, dtype='float64'))
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)

    print('The no corruption code for file %s ran for %.2fm' % (os.path.split(__file__)[1], training_time / 60.),
          file=sys.stderr)
    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28),
        tile_shape=(20, 20),
        tile_spacing=(1, 1)
    ))
    image.save('filters_corruption_0_2.png')

    print('build model 2 ...')
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = DeNosingAutoEncoder(np_rng=rng, theano_rng=theano_rng, x_in=x, n_visible=28 * 28, n_hidden=500)
    cost, updates = da.get_cost_updates(corruption_level=0.3, learning_rate=learning_rate)
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()
    print('training 2...')
    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print('Training epoch %d, cost ' % epoch, np.mean(c, dtype='float64'))
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)

    print('The 30%% corruption code for file %s ran for %.2fm' % (os.path.split(__file__)[1], training_time / 60.),
          file=sys.stderr)
    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28),
        tile_shape=(20, 20),
        tile_spacing=(1, 1)
    ))
    image.save('filters_corruption_30_2.png')
    os.chdir('../')

if __name__ == '__main__':
    test_dA()
