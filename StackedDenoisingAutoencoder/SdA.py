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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from LogisticRegression.utils import load_data
from LogisticRegression.logistic_msgd import LogisticRegression
from MultilayerPerceptron.mlp import HiddenLayer
from DenoisingAutoencoder.dA import DeNosingAutoEncoder


class StackedDeNosingAutoEncoder(object):
    def __init__(self, np_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=None, n_outs=10, corruption_levels=None):
        """
        参数意义参照DenoisingAutoencoder.dA.DeNosingAutoEncoder
        """
        if hidden_layers_sizes is None:
            hidden_layers_sizes = [500, 500]
        if corruption_levels is None:
            corruption_levels = [0.1, 0.1]

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in range(self.n_layers):
            # 确定参数和数据
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.sigmoid_layers[-1].output

            # 构造隐藏层
            sigmoid_layer = HiddenLayer(rng=np_rng,
                                        x_in=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # 构造自编码层，和隐藏层权重共享
            dA_layer = DeNosingAutoEncoder(np_rng=np_rng,
                                           theano_rng=theano_rng,
                                           x_in=layer_input,
                                           n_visible=input_size,
                                           n_hidden=hidden_layers_sizes[i],
                                           W=sigmoid_layer.W,
                                           bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        # 最后添加LR层
        self.logLayer = LogisticRegression(
            x_in=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )
        self.params.extend(self.logLayer.params)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)  # 微调用到的cost
        self.errors = self.logLayer.errors(self.y)  # 错误率输出

    def pretraining_functions(self, train_set_x, batch_size):
        """
        预训练函数，预训练：先进行去噪自编码训练数据，得到两个hidden layers的权重

        :param train_set_x: Shared variable that contains all datapoints used for training the dA
        :param batch_size: size of a [mini]batch
        """

        index = T.lscalar('index')  # index to a [mini]batch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        batch_begin = index * batch_size  # begining of a batch, given `index`
        batch_end = batch_begin + batch_size  # ending of a batch given `index`

        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level, learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level, value=0.2),
                    theano.In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """
        确立微调函数：使用自编码训练后，再用LR函数进行反向传播进行自编码的参数和LR参数共同调整

        :param datasets: It is a list that contain all the datasets
        :param batch_size: size of a minibatch
        :param learning_rate: learning rate used during finetune stage
        """

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        gparams = T.grad(self.finetune_cost, self.params)  # compute the gradients with respect to the model parameters
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]  # compute list of fine-tuning updates

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score


def test_SdA(finetune_lr=0.1, pretraining_epochs=15, pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.
    This is demonstrated on MNIST.

    :param finetune_lr: learning rate used in the finetune stage (factor for the stochastic gradient)
    :param pretraining_epochs: number of epoch to do pretraining
    :param pretrain_lr: learning rate to be used during pre-training
    :param training_epochs: maximal number of iterations ot run the optimizer
    :param dataset: path the the pickled dataset
    :param batch_size: number of batch size
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    np_rng = np.random.RandomState(89677)
    print('... building the model')
    sda = StackedDeNosingAutoEncoder(
        np_rng=np_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[1000, 1000, 1000],
        n_outs=10
    )

    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    corruption_levels = [.1, .2, .3]
    for i in range(sda.n_layers):
        for epoch in range(pretraining_epochs):
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c, dtype='float64')))

    end_time = timeit.default_timer()

    print('The pretraining code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_time-start_time)/60.),
          file=sys.stderr)

    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    print('... finetunning the model')
    patience = 10 * n_train_batches
    patience_increase = 2.
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            train_fn(minibatch_index)
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter_num

                    test_losses = test_model()
                    test_score = np.mean(test_losses, dtype='float64')
                    print('     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
            if patience <= iter_num:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print(('The training code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_time-start_time)/60.)),
          file=sys.stderr)


if __name__ == '__main__':
    test_SdA()
