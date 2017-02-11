#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

embeddings = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nv+1, de)).astype(theano.config.floatX))