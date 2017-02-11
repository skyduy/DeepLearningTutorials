#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""
from __future__ import print_function
import sys
import time
from collections import OrderedDict

import theano
import numpy as np
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from optimizer import optimizer
from utils import ortho_weight, numpy_floatX
from utils import load_data, prepare_data, get_minibatches_idx


class LSTM(object):
    def __init__(self, n_word, h_dim, n_out, dropout=False, dropout_seed=123, noise=0.5):
        """
        :param n_word: 词汇量
        :param h_dim: 单词向量长度
        :param n_out: 待分类个数
        :param dropout: 是否使用dropout
        :param dropout_seed: 使用dropout时，指定种子
        :param noise : 使用dropout且利用噪音时，噪音比重
        """
        self.n_word = n_word
        self.h_dim = h_dim
        self.n_out = n_out
        self.use_dropout = dropout
        self.dropout_SEED = dropout_seed
        self.noise = noise

        self.word_embedding = theano.shared(
            (0.01 * np.random.rand(self.n_word, self.h_dim)).astype(theano.config.floatX),  # word embedding
            name='word_embedding',
        )

        self.cell_W = theano.shared(
            np.concatenate(
                [ortho_weight(self.h_dim), ortho_weight(self.h_dim),
                 ortho_weight(self.h_dim), ortho_weight(self.h_dim)],
                axis=1
            ),
            name='cell_W',
        )

        self.cell_U = theano.shared(
            np.concatenate(
                [ortho_weight(self.h_dim), ortho_weight(self.h_dim),
                 ortho_weight(self.h_dim), ortho_weight(self.h_dim)],
                axis=1
            ),
            name='cell_U',
        )

        self.cell_b = theano.shared(
            np.zeros((4 * self.h_dim,)).astype(theano.config.floatX),
            name='cell_b',
        )

        self.U = theano.shared(
            0.01 * np.random.randn(self.h_dim, self.n_out).astype(theano.config.floatX),
            name='U',
        )
        self.b = theano.shared(
            np.zeros((self.n_out,)).astype(theano.config.floatX),
            name='b',
        )

        self.params = [
            self.word_embedding,
            self.cell_W, self.cell_U, self.cell_b,
            self.U, self.b,
        ]

    def build_model(self):
        """
        :return: (use_noise, x, mask, y, cost, f_pred)
            use_noise: 指定是否使用noise进行dropout，dropout有效时，该参数有效
            x: 数据的符号
            mask: 数据掩码符号
            y: 标签符号
            cost: 代价结果
            f_pred: 计算预测的函数

        """
        use_noise = theano.shared(numpy_floatX(0.))
        x = T.matrix('x', dtype='int64')
        y = T.vector('y', dtype='int64')
        mask = T.matrix('mask', dtype=theano.config.floatX)

        word_number = x.shape[0]  # 单词数
        sentence_number = x.shape[1]  # 句子个数
        emb = self.word_embedding[x.flatten()].reshape([word_number, sentence_number, self.h_dim])  # word2vec

        last_features = self.after_lstm(emb, mask, use_noise)  # 获取最终features: (sentence_number, ndim)

        p_y_given_x = T.nnet.softmax(T.dot(last_features, self.U) + self.b)
        off = 1e-8
        if p_y_given_x.dtype == 'float16':
            off = 1e-6
        cost = -T.log(p_y_given_x[T.arange(sentence_number), y] + off).mean()

        # 创建计算预测结果的函数
        f_pred = theano.function([x, mask], p_y_given_x.argmax(axis=1), name='f_pred')
        return use_noise, x, mask, y, cost, f_pred

    def after_lstm(self, state_below, mask, use_noise):
        """
        :param state_below:  初始vec数据
        :param mask:  对应单词掩码
        :param use_noise:  使用dropout时指定是否使用噪声
        :return:
        """
        word_numbers = state_below.shape[0]
        if state_below.ndim == 3:
            sentence_number = state_below.shape[1]
        else:
            sentence_number = 1

        # 辅助函数： 切片
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        # 一次性求出初始的input gate, candidate value, forget gate和output gate，不带权重U
        # shape form (word_number, sentence_num, ndim) to (word_number, sentence_number, 4*ndim)
        state_below = (T.dot(state_below, self.cell_W) + self.cell_b)

        def recurrence(m_, x_, h_, c_):
            """
            :param m_: mask, 后面会转化为shape (sentence_number, 1), 元素由0或1组成，表示该次的单词是否有效
            :param x_: shape (sentence_number, 4*h_dim), 要被处理的新信息
            :param h_: shape (sentence_number, h_dim) 上层信息
            :param c_: shape (sentence_number, h_dim) 上层原胞
            :return:  本层输出信息和原胞
            """
            preact = T.dot(h_, self.cell_U)  # shape (sentence_number, 4*ndim))
            preact += x_

            #  以下四个shape均为(sentence_number, ndim)
            i = T.nnet.sigmoid(_slice(preact, 0, self.h_dim))  # 获取input门
            f = T.nnet.sigmoid(_slice(preact, 1, self.h_dim))  # 获取forget门
            o = T.nnet.sigmoid(_slice(preact, 2, self.h_dim))  # 获取output门
            c = T.tanh(_slice(preact, 3, self.h_dim))  # 获取candidate value

            # 获取本层原胞，和输出信息
            c = f * c_ + i * c
            h = o * T.tanh(c)

            # 若某个句子到头，保留上层原胞/输出信息，否则保留本层原胞/输出信息
            c = m_[:, None] * c + (1. - m_)[:, None] * c_
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            # 返回本层输出信息和原胞
            return h, c

        hidden_cells, updates = theano.scan(
            recurrence,
            sequences=[
                mask,  # shape: (word_number, sentence_number)
                state_below  # shape: (word_number, sentence_number, 4*ndim)
            ],
            outputs_info=[
                T.alloc(numpy_floatX(0.), sentence_number, self.h_dim),  # 初始的上层输入
                T.alloc(numpy_floatX(0.), sentence_number, self.h_dim)   # 初始的上层原胞
            ],
            name='scan',
            n_steps=word_numbers
        )

        # 所有单词输出 shape(word_number, sentence_number, ndim)
        all_hidden = hidden_cells[0]
        # 先去除无效单词的特征，再将每个句子各有效单词特征求平均 shape(sentence_num, ndim)
        features = (all_hidden * mask[:, :, None]).sum(axis=0) / mask.sum(axis=0)[:, None]

        # 若使用dropout
        if self.use_dropout:
            trng = MRG_RandomStreams(self.dropout_SEED)
            features = T.switch(
                use_noise,  # 若使用noise
                (features * trng.binomial(features.shape, p=1-self.noise, n=1, dtype=features.dtype)),
                features * (1-self.noise)
            )

        return features

    def get_params(self):
        results = OrderedDict()
        for p in self.params:
            results[p.name] = p.get_value()
        return results

    def set_params(self, params):
        pd = OrderedDict()
        for p in self.params:
            pd[p.name] = p
        for k, v in params.iteritems():
            pd[k].set_value(v)


def pred_correct(f_pred, f_prepare_data, data, iterator):
    """
    计算预测准确率
    :param f_pred: 获取预测结果的函数
    :param f_prepare_data:  数据预处理函数
    :param data:  原始数据
    :param iterator:  数据索引信息
    :return:
    """
    valid_correct = 0
    for _, valid_index in iterator:
        x, mask, y = f_prepare_data([data[0][t] for t in valid_index], np.array(data[1])[valid_index], maxlen=None)
        preds = f_pred(x, mask)
        targets = np.array(data[1])[valid_index]
        valid_correct += (preds == targets).sum()
    return numpy_floatX(valid_correct) / len(data[0])


def train_lstm(n_words=10000, h_dim=128, dropout=True, noise=0.5, maxlen=100, decay_c=0., opt_name='adadelta',
               lrate=0.0001, max_epochs=5000, patience=10, batch_size=16, valid_freq=370, valid_batch_size=64,
               test_size=-1, disp_freq=10, save_freq=1110, f_best_model='model/lstm_model.npz'):
    """
    初始化lstm的参数：
    :param n_words: 词汇量，和h_dim共同构造 word embedding
    :param h_dim: 词向量维度 和 LSTM单元中隐藏层数量
    :param dropout:  LSTM的最后一层softmax是否使用dropout
    :param noise: 在使用dropout且是加噪方式的drop时，指定噪音比重

    数据加载参数：
    :param maxlen: 允许最大句子长度

    代价函数即其优化相关参数：
    :param decay_c:  权重衰减比重
    :param opt_name: 优化方式, 必须为['sgd', 'adadelta', 'rmsprop']中的一种
    :param lrate: Learning rate for sgd (For adadelta and rmsprop, it's initial weight)

    early-stop训练参数：
    :param max_epochs: 最大训练epochs
    :param patience: early stop参数
    :param batch_size: 训练的batch大小
    :param valid_freq: 验证频率
    :param valid_batch_size: valid的batch大小
    :param test_size: 测试集大小

    其它参数：
    :param disp_freq: 展示训练结果频率
    :param save_freq: 保存频率
    :param f_best_model: 保存位置

    :return: train_corr, valid_corr, test_corr
    """

    print('Loading data...')
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    # 此时test集是按照句子长度排好的，因此进行随机打乱并抽取
    if test_size > 0:
        idx = np.arange(len(test[0]))
        np.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
    ydim = np.max(train[1]) + 1  # 获取label数量

    print('Building model...')
    # 构造LSTM模型
    lstm = LSTM(n_words, h_dim, ydim, dropout, noise=noise)

    # 获取需要的自变量和因变量和函数
    use_noise, x, mask, y, cost, f_pred = lstm.build_model()
    if decay_c > 0.:  # 更新cost
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (lstm.U ** 2).sum()  # L2 regularization
        weight_decay *= decay_c
        cost += weight_decay
    grads = T.grad(cost, wrt=lstm.params)  # 获取梯度
    lr = T.scalar(name='lr')

    # 获取两个函数，第一个函数用来获取cost和梯度更新值。第二个函数用来更新梯度。
    f_grad_shared, f_update = optimizer(opt_name, lr, lstm.params, grads, x, mask, y, cost)

    print('Training...')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    # 训练参数
    history_corrs = []  # 历史正确率
    best_p = None
    bad_counter = 0
    update_times = 0
    early_stop = False
    epoch = 0
    if valid_freq == -1:
        valid_freq = len(train[0]) // batch_size
    if save_freq == -1:
        save_freq = len(train[0]) // batch_size

    start_time = time.time()
    try:
        for epoch in range(max_epochs):
            n_samples = 0
            # 每个epoch均将原数据打乱
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                update_times += 1
                use_noise.set_value(1.)  # 设置变量use_noise为True

                # 准备数据
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                # 本轮后续处理
                if np.isnan(cost) or np.isinf(cost):
                    print('Bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(update_times, disp_freq) == 0:
                    print('Epoch ', epoch, 'Update ', update_times, 'Cost ', cost)

                if f_best_model and np.mod(update_times, save_freq) == 0:
                    print('Saving...')
                    if best_p is not None:
                        params = best_p
                    else:
                        params = lstm.get_params()
                    np.savez(f_best_model, history_errs=history_corrs, **params)
                    print('Saving done')

                if np.mod(update_times, valid_freq) == 0:
                    use_noise.set_value(0.)  # 在进行validate时, 确保原数据输入
                    train_corr = pred_correct(f_pred, prepare_data, train, kf)
                    valid_corr = pred_correct(f_pred, prepare_data, valid, kf_valid)
                    test_corr = pred_correct(f_pred, prepare_data, test, kf_test)

                    history_corrs.append([valid_corr, test_corr])

                    if best_p is None or valid_corr <= np.array(history_corrs)[:, 0].min():
                        best_p = lstm.get_params()
                        bad_counter = 0

                    print('Train ', train_corr, 'Valid ', valid_corr, 'Test ', test_corr)

                    if len(history_corrs) > patience and valid_corr >= np.array(history_corrs)[:-patience, 0].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            early_stop = True
                            break

            print('Seen %d sentence' % n_samples)

            if early_stop:
                break

    except KeyboardInterrupt:
        print("Training interupted")
    end_time = time.time()

    # 加载最好的模型参数
    if best_p is not None:
        lstm.set_params(best_p)
    else:
        best_p = lstm.get_params()

    # 最终在最好的模型上测试一发
    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_corr = pred_correct(f_pred, prepare_data, train, kf_train_sorted)
    valid_corr = pred_correct(f_pred, prepare_data, valid, kf_valid)
    test_corr = pred_correct(f_pred, prepare_data, test, kf_test)

    print('Train ', train_corr, 'Valid ', valid_corr, 'Test ', test_corr)
    if f_best_model:
        np.savez(f_best_model, train_err=train_corr, valid_err=valid_corr, test_err=test_corr,
                 history_errs=history_corrs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (epoch+1, (end_time-start_time)/(1.*(epoch+1))))
    print(('Training took %.1fs' % (end_time - start_time)), file=sys.stderr)
    return train_corr, valid_corr, test_corr


if __name__ == '__main__':
    train_lstm(max_epochs=100, test_size=500,)
