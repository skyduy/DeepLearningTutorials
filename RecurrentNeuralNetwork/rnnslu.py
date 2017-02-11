#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

from __future__ import print_function

from collections import OrderedDict
import os
import copy
import random
import timeit
import subprocess

import theano
import numpy as np
from theano import tensor as T
from utils import atisfold, context_window, shuffle, conlleval

# the instance of RNNSLU an extremely deep structure. Otherwise deepcopy failed
import sys
sys.setrecursionlimit(1500)


class SimpleRNN(object):
    """
    递归神经网络的一种实现： elman neural net model
    wikipedia: Elman and Jordan networks are also known as "simple recurrent networks" (SRN).
    """
    def __init__(self, nh, nc, ne, de, cs):
        """
        :param nh: dimension of the hidden layer
        :param nc: number of classes
        :param ne: number of word embeddings in the vocabulary 即总单词数目
        :param de: dimension of the word embeddings 即每个单词的用多少分量重构（类似于非负矩阵分解）
        :param cs: word window context size
        """

        # 为每个单词的构造向量。 其大小多出一行是为了PADDING，PADDING的作用见函数context_window
        # 初始未球化原因：1、仅作为参数看待  2、影响不大
        self.emb = theano.shared(
            name='embeddings',
            value=0.2 * np.random.uniform(-1.0, 1.0, (ne+1, de)).astype(theano.config.floatX)
        )

        # 输入层的至隐藏层权重
        self.wx = theano.shared(
            name='wx',
            value=0.2 * np.random.uniform(-1.0, 1.0, (de * cs, nh)).astype(theano.config.floatX)
        )

        # 上级隐藏层至本级隐藏层权重
        self.wh = theano.shared(
            name='wh',
            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX)
        )
        # 上级隐藏层的偏移
        self.bh = theano.shared(name='bh', value=np.zeros(nh, dtype=theano.config.floatX))
        # 初始“上级隐藏层”，即第一个隐藏层的上级隐藏层
        self.h0 = theano.shared(name='h0', value=np.zeros(nh, dtype=theano.config.floatX))

        # 最终隐藏层全连接权重
        self.w = theano.shared(
            name='w',
            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nc)).astype(theano.config.floatX)
        )
        # 最终隐藏层全连接偏移
        self.b = theano.shared(name='b', value=np.zeros(nc, dtype=theano.config.floatX))

        # bundle： 参数束
        self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.h0]

        # idxs的shape为 (context_windows.size, words_number_in_sentence)
        # self.emb[idxs]作用是将x构造为和idxs同shape的matrix，x中相应于idxs位置的元素变为emb的行（每个元素均为emb的某一行）
        # 会将每个单词转化为带有上下文单词的向量表示
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))

        y_sentence = T.ivector('y_sentence')  # labels

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)  # (1, nh) 1行nh列
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)  # (1, nc) 1行nc列， 即[[P1, P2, ..., Pnc]]
            return [h_t, s_t]

        """
        逐步扫描句子中所有单词

        scan 此处讲解：
            首先，recurrence函数接收两个参数，第一个x_t从sequences中顺序取，第二个从outputs_info中取非None的那个
            而outputs_info的限制为其和fn返回的shape一致，所以，非None的那个作用之一是指定初始值，
            作用二便是接下来选取函数fn返回的哪一个。
            所以，在这里，fn返回的h_t会作为下一次的h_tm1传入，以此类推。
            n_steps会限制fn的调用次数，所以这里n_steps加上没有作用，因为默认就能调用完x
            （句子的长短决定scan次数）
        """
        [h, s], _ = theano.scan(
            fn=recurrence,
            sequences=x,
            outputs_info=[self.h0, None],
            # n_steps=x.shape[0]  # 此句可有可无
        )

        # recurrence函数返回的第二个变量shape为(1, nc), 迭代了n_steps次，故此s的shape为(n_steps, 1, nc)
        p_y_given_x_sentence = s[:, 0, :]
        # argmax axis=1求每一行的最大值下标。
        # sum axis=1 求每一行的和。
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # 计算代价值，梯度和更新列表
        lr = T.scalar('lr')
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, sentence_gradients))

        # 训练函数，每次接收一个句子进行训练
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        self.normalize = theano.function(
            inputs=[],
            # dimshuffle作用保证在行上进行除。确定正确性
            updates={self.emb: self.emb / T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')}
        )

        # 分类函数，会对每个句子的每个单词进行分类，即每个句子是一个mini-batch
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

    def train(self, x, y, window_size, learning_rate):
        cwords = context_window(x, window_size)
        words = list(map(lambda each: np.asarray(each).astype('int32'), cwords))
        labels = y

        self.sentence_train(words, labels, learning_rate)
        self.normalize()

    def save(self, folder):
        for param in self.params:
            np.save(os.path.join(folder, param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(np.load(os.path.join(folder, param.name + '.npy')))


def test_rnnslu(param=None):
    if param is None:
        param = {
            'fold': 3,  # folds in 0,1,2,3,4
            'data': 'atis',
            'lr': 0.0970806646812754,
            'verbose': 1,
            'decay': True,  # decay on the learning rate if improvement stops
            'win': 7,  # number of words in the context window
            'nhidden': 200,  # number of hidden units
            'seed': 345,
            'emb_dimension': 50,  # dimension of word embedding
            'nepochs': 60,  # 60 is recommended
            'savemodel': True
        }

    folder_name = os.path.basename(__file__).split('.')[0]
    folder = os.path.join(os.path.dirname(__file__), folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    train_set, valid_set, test_set, dic = atisfold(param['fold'])

    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    # 可视的真实数据，用于效果比较和F1-Score计算
    groundtruth_valid = [map(lambda item: idx2label[item], y) for y in valid_y]
    words_valid = [map(lambda item: idx2word[item], w) for w in valid_lex]
    groundtruth_test = [map(lambda item: idx2label[item], y) for y in test_y]
    words_test = [map(lambda item: idx2word[item], w) for w in test_lex]

    # 初始化模型
    np.random.seed(param['seed'])
    random.seed(param['seed'])
    rnn = SimpleRNN(nh=param['nhidden'],
                    nc=nclasses,
                    ne=vocsize,
                    de=param['emb_dimension'],
                    cs=param['win'])

    # 训练
    best_f1 = -np.inf
    best_rnn = None
    param['clr'] = param['lr']
    for e in range(param['nepochs']):
        # 每个epoch以同样方式打乱mini-batch顺序，亦即句子顺序
        shuffle([train_lex, train_ne, train_y], param['seed'])

        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            # 每次将一个句子推入模型进行训练
            rnn.train(x, y, param['win'], param['clr'])
        print('[learning] epoch %i completed in %.2f (sec)' % (e, timeit.default_timer() - tic))

        # 将模型应用于数据，并输出可视的预测label
        predictions_test = [
            map(lambda each: idx2label[each], rnn.classify(np.asarray(context_window(x, param['win'])).astype('int32')))
            for x in test_lex
        ]
        predictions_valid = [
            map(lambda each: idx2label[each], rnn.classify(np.asarray(context_window(x, param['win'])).astype('int32')))
            for x in valid_lex
        ]

        # 打印预测标签、真实标签、对应单词，并获取f1值
        res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt', folder)
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt', folder)

        if res_valid['f1'] > best_f1:
            if param['savemodel']:
                rnn.save(folder)

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid['f1']
            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'valid F1', res_valid['f1'],
                      'best test F1', res_test['f1'])

            param['vf1'], param['tf1'] = res_valid['f1'], res_test['f1']
            param['vp'], param['tp'] = res_valid['p'], res_test['p']
            param['vr'], param['tr'] = res_valid['r'], res_test['r']
            param['be'] = e
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
          'valid F1', param['vf1'],
          'best test F1', param['tf1'],
          'with the model', folder)


if __name__ == '__main__':
    test_rnnslu()
