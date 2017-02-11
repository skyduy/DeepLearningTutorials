#!/usr/bin/python
# coding: utf-8

"""
    Restricted Boltzmann Machine (RBM)
    This tutorial introduces restricted boltzmann machines (RBM) using Theano.

    Boltzmann Machines (BMs) are a particular form of energy-based model which
    contain hidden variables. Restricted Boltzmann Machines further restrict BMs
    to those without visible-visible and hidden-hidden connections.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

from __future__ import print_function

import os
import timeit
import numpy as np
import theano
import theano.tensor as T
from PIL import Image
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from LogisticRegression.utils import load_data
from DenoisingAutoencoder.utils import tile_raster_images


class RBM(object):
    def __init__(self, x_in=None, n_visible=784, n_hidden=500, W=None, hbias=None, vbias=None,
                 np_rng=None, theano_rng=None):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param x_in: None for standalone RBMs or symbolic variable if RBM is part of a larger graph.
        :param n_visible: number of visible units
        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
                  shared weight matrix in case RBM is part of a DBN network; in a DBN,
                  the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
                      to a shared hidden units bias vector in case RBM is part of a
                      different network

        :param vbias: None for standalone RBMs or a symbolic variable
                      pointing to a shared visible units bias

        :param np_rng:
        :param theano_rng:
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if np_rng is None:
            np_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if W is None:
            initial_W = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            hbias = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = x_in
        if not x_in:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        """
        RBM能量函数的自由能形式
        引入自由能的原因是，对RBM服对数求导结果可转化为自由能求导形式，简化运算
        """
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def sample_h_given_v(self, vis):
        """
        先从可视层各节点的‘概率值’通过参数得出隐藏层个节点的‘概率值’，再根据概率值进行抽样（为Gibbs抽样的步骤）。
        因为在计算时，概率值结果的表示刚好为sigmoid函数，而该函数又是一种激励函数，因此才把RBM也叫做一种神经网络模型。

        :param vis: 可视层节点单元
        :return: [概率激活前的值，激活后的概率值，抽样值]
        """
        # 因为在后面还需要用到pre_active，为了theano效率，这里保存一个模型
        pre_active = T.dot(vis, self.W) + self.hbias
        after_active = T.nnet.sigmoid(pre_active)
        h1_sample = self.theano_rng.binomial(size=after_active.shape, n=1, p=after_active, dtype=theano.config.floatX)
        return [pre_active, after_active, h1_sample]

    def sample_v_given_h(self, hid):
        """
        从隐藏层获取可视层的抽样，tied weight
        """
        pre_active = T.dot(hid, self.W.T) + self.vbias
        after_active = T.nnet.sigmoid(pre_active)
        v1_sample = self.theano_rng.binomial(size=after_active.shape, n=1, p=after_active, dtype=theano.config.floatX)
        return [pre_active, after_active, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """
        从隐藏态样本获取可视态抽样，再获取隐藏态样本
        """
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """
        从可视态样本获取隐藏态抽样，再获取可视态样本
        """
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """
        CN-k 或 PCD-k 的步骤。训练时会不断调用以获取参数的update。

        所有分布都可以通过模拟马尔科夫过程，经过足够多的转移后，充分接近平稳分布。
        接近平稳分布时，隐藏层的分布便和可视层分布大概一致，这里也是需要提取隐藏层和可视层的参数。

        为了调整参数，每次收敛需要大量步骤和数据。而CD算法可以每K次收敛快速调整参数，最终达到效果。

        :param lr: learning rate used to train the RBM
        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).
        :param k: number of Gibbs steps to do in CD-k/PCD-k

        :returns 评估代价值，参数更新值
        """

        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)  # 获取第一步隐藏层的抽样
        if persistent is None:
            chain_start = ph_sample  # for CD, use the newly generate hidden sample
        else:
            chain_start = persistent  # for PCD, initialize from the old state of the chain

        (
            [
                pre_sigmoid_nvs, nv_means, nv_samples,
                pre_sigmoid_nhs, nh_means, nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],  # 前一步输出结果的初始状态
            n_steps=k,  # 迭代次数
            name="gibbs_hvh"
        )
        chain_end = nv_samples[-1]  # 获取K次抽样后的可视态节点

        # 此处代价不是自由能的代价，只不过可以使用自由能简化运算。
        # 该代价使变换后的分布尽可能与原数据的分布一致
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        # chain_end是模型参数项中的一个的象征性的Theano变量
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        if persistent:
            # PCD中要更新本次结束时的 Gibbs sampling chain
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        # 上面的cost是来形容隐藏层和可视层的分布相似度，此处cost是用来评估RBM的
        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        # TODO 不懂为什么PCA可以用这个来衡量。根本原因还是自由能理解不透彻。
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        print(bit_i_idx.get_value())
        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # 更改params时，也会顺带更改bit_i_idx以保证各下标均匀获取
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error"""

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +  # 为1的部分乘以为1的概率
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),  # 为0的部分乘以为0的概率的
                axis=1
            )
        )

        return cross_entropy


def test_rbm(learning_rate=0.1, training_epochs=15,  batch_size=20, n_chains=20, n_samples=10,
             n_hidden=500, output_folder='rbm_plots', dataset='mnist.pkl.gz'):
    """
    Demonstrate how to train and afterwards sample from it using Theano.
    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    :param n_hidden: number of hidden units
    :param output_folder: output folder
    :param dataset: path the the pickled dataset
    :return:
    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    index = T.lscalar()
    x = T.matrix('x')
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    persistent_chain = theano.shared(
        np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True
    )  # 初始化持久链为 np.zeros((batch_size, n_hidden)
    rbm = RBM(x_in=x, n_visible=28 * 28, n_hidden=n_hidden, np_rng=rng, theano_rng=theano_rng)

    # 通过15-PCD获取每次的模型估计和参数更新表
    cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=15)

    print('Training the RBM...')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]},
        name='train_rbm'
    )  # 训练RBM以获取合适的转换参数
    plotting_time = 0.
    start_time = timeit.default_timer()
    for epoch in range(training_epochs):
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        print('Training epoch %d, cost is ' % epoch, np.mean(mean_cost))
        plotting_start = timeit.default_timer()
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )  # 每个epoch均绘制一张权重图片
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)
    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time) - plotting_time
    print ('Training took %f minutes' % (pretraining_time / 60.))

    print('Sampling from the RBM...')
    # 随机获取 n_chains 个测试样本进行抽样
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(np.asarray(
        test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains], dtype=theano.config.floatX
    ))

    plot_every = 1000  # 不断绘制1000次后绘一次图
    (
        [
            presig_hids, hid_mfs, hid_samples,
            presig_vis, vis_mfs, vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name="gibbs_vhv"
    )
    # 因为使用PCD，所以在测试中也要保存上次的 Gibbs sampling chain，为了第二次scan时初始outputs_info使用了持久链尾值
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # 将RBM过程的updates应用至函数
    sample_fn = theano.function(
        [],
        [vis_mfs[-1], vis_samples[-1]],
        updates=updates,
        name='sample_fn'
    )

    # 构造存储空间
    image_data = np.zeros((29*n_samples+1, 29*n_chains-1), dtype='uint8')
    for idx in range(n_samples):
        vis_mf, vis_sample = sample_fn()  # 此时的updates.update语句生效，使用持久对比散度（每plot_every次绘一次）
        print(' ... plotting sample %d' % idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )
    image = Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../')

if __name__ == '__main__':
    test_rbm()
