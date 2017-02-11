#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""


import theano
import theano.tensor as T
from utils import numpy_floatX


def optimizer(name, lr, params, grads, x, mask, y, cost):
    assert name in ['sgd', 'adadelta', 'rmsprop']
    if name == 'sgd':
        return sgd(lr, params, grads, x, mask, y, cost)
    elif name == 'adadelta':
        return adadelta(lr, params, grads, x, mask, y, cost)
    else:
        return rmsprop(lr, params, grads, x, mask, y, cost)


def sgd(lr, params, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """

    # New set of shared variable that will contain the gradient for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name=p.name) for p in params]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, name='sgd_f_grad_shared')

    # Function that updates the weights from the previously computed gradient.
    pup = [(p, p - lr * g) for p, g in zip(params, gshared)]
    f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, params, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    :param lr: Initial learning rate
    :param params: Model parameters
    :param grads: Gradients of cost w.r.t to parameres
    :param x: Model inputs
    :param mask: Sequence mask
    :param y: Targets
    :param cost: Objective fucntion to minimize
    :return:
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % p.name) for p in params]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % p.name) for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % p.name) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(params, updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, params, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    :param lr: Initial learning rate
    :param params: Model parameters
    :param grads: Gradients of cost w.r.t to parameres
    :param x: Model inputs
    :param mask: Sequence mask
    :param y: Targets
    :param cost: Objective fucntion to minimize
    :return:
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % p.name) for p in params]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad' % p.name) for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % p.name) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rgup + rg2up, name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_updir' % p.name) for p in params]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(params, updir_new)]
    f_update = theano.function([lr], [],
                               updates=updir_new + param_up, on_unused_input='ignore', name='rmsprop_f_update')

    return f_grad_shared, f_update
