#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

from __future__ import print_function, division
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from LogisticRegression.utils import load_data
from LogisticRegression.logistic_msgd import LogisticRegression
from MultilayerPerceptron.mlp import HiddenLayer
from RestrictedBoltzmannMachine.rbm import RBM


class DBN(object):
    """
    Deep Belief Network

    先用DBN提取特征，再进行LR(微调)
    """

    def __init__(self, np_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=None, n_outs=10):
        """
        This class is made to support a variable number of layers.

        :param np_rng: np random number generator used to draw initial weights
        :param theano_rng: Theano random generator; if None is given one is generated based on a seed drawn from `rng`
        :param n_ins: dimension of the input to the DBN
        :param hidden_layers_sizes: intermediate layers size, must contain at least one value
        :param n_outs: dimension of the output of the network
        """
        if hidden_layers_sizes is None:
            hidden_layers_sizes = [500, 500]
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(np_rng.randint(2 ** 30))

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        # 此处处理方式类似StackedDenoisingAutoencoder
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

            # 构造RBM层，和隐藏层权重共享
            rbm_layer = RBM(np_rng=np_rng,
                            theano_rng=theano_rng,
                            x_in=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # 最后添加LR层，用于Fine tuning
        self.logLayer = LogisticRegression(
            x_in=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)  # 微调用到的cost
        self.errors = self.logLayer.errors(self.y)  # 错误率输出

    def pretraining_functions(self, train_set_x, batch_size, k):
        """
        预训练函数，预训练：进行多层RBM训练（亦即DBN)，此处得到两个hidden layers的权重

        :param train_set_x: Shared variable that contains all datapoints used for training the dA
        :param batch_size: size of a [mini]batch
        :param k: 每步CD-k的抽取次数
        """

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('lr')  # learning rate to use
        batch_begin = index * batch_size  # begining of a batch, given `index`
        batch_end = batch_begin + batch_size  # ending of a batch given `index`

        pretrain_fns = []
        for rbm in self.rbm_layers:
            cost, updates = rbm.get_cost_updates(learning_rate, persistent=None, k=k)
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """
        确立微调函数：使用DBN训练后，再用LR函数进行反向传播进行DBN得到的参数和LR参数共同调整

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


def test_DBN(finetune_lr=0.1, pretraining_epochs=100, pretrain_lr=0.01, k=1, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=10):
    """
    Demonstrates how to train and test a Deep Belief Network.
    This is demonstrated on MNIST.

    :param finetune_lr: learning rate used in the finetune stage (factor for the stochastic gradient)
    :param pretraining_epochs: number of epoch to do pretraining
    :param pretrain_lr: learning rate to be used during pre-training
    :param k: number of Gibbs steps in CD/PCD
    :param training_epochs: maximal number of iterations ot run the optimizer
    :param dataset: path the the pickled dataset
    :param batch_size: number of batch size
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    np_rng = np.random.RandomState(123)
    print('... building the model')
    dbn = DBN(np_rng=np_rng, n_ins=28 * 28, hidden_layers_sizes=[1000, 1000, 1000], n_outs=10)

    print('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size, k=k)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    for i in range(dbn.n_layers):
        for epoch in range(pretraining_epochs):
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
            print(np.mean(c, dtype='float64'))

    end_time = timeit.default_timer()
    print('The pretraining code for file %s ran for %.2fm' % (os.path.split(__file__)[1], (end_time-start_time)/60.),
          file=sys.stderr)

    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetuning the model')
    patience = 4 * n_train_batches
    patience_increase = 2.
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)
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
                    if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
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
    print(('Optimization complete with best validation score of %f %%, '
           'obtained at iteration %i, '
           'with test performance %f %%'
           ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The fine tuning code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)


if __name__ == '__main__':
    test_DBN()
