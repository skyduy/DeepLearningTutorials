#!/usr/bin/python
# coding: utf-8

"""
    ...
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Author: YuJun <cuteuy@gmail.com>
"""

import os
import stat
import urllib
import random
import cPickle
import subprocess
import numpy as np
from pprint import pprint


def shuffle(lol, seed):
    """
    shuffle inplace each list in the same order

    :param lol: list of list as input
    :param seed: seed the shuffling
    :return:
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def context_window(l, win):
    """
    使用效果见 test_context_windows 输出
    :param l: array containing the word indexes
    :param win: int corresponding to the size of the window given a list of indexes composing a sentence
    :return: a list of list of indexes corresponding to context windows surrounding each word in the sentence
    """
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    # 超出单词index边界的都指向最后的通用padding，padding见CNN __init__函数
    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def atisfold(fold):
    """
    data loading functions
    :param fold:
    :return : (train_set, valid_set, test_set, dicts)
        train_set: (words, tables, labels), in which each is an array.
        valid_set: (words, tables, labels), in which each is an array.
        test_set: (words, tables, labels), in which each is an array.
        dicts: 用下表链接各属性的字典
            {
                'labels2idx': {'B-aircraft_code': 0, 'B-airline_code': 1, 中间省略, 'I-transport_type': 125, 'O': 126},
                'tables2idx': {'<NOTABLE>': 0, 'B-about': 1, 'B-after': 2, 中间省略, 'month_name': 139, 'time_mod': 140},
                'words2idx': {"'d": 0, 省略 "'t": 5, '72s': 6, '<UNK>': 7, 'DIGIT': 8, 省略, 'you': 570, 'your': 571}
            }
    """
    assert fold in range(5)
    prefix = os.getenv('ATISDATA', os.path.join(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0], 'data'))
    filename = os.path.join(prefix, 'atis.fold'+str(fold)+'.pkl')

    try:
        train_set, valid_set, test_set, dicts = cPickle.load(open(filename))
        return train_set, valid_set, test_set, dicts
    except Exception as e:
        print('Error occurred:', e.message)


def atisfull():
    prefix = os.getenv('ATISDATA', os.path.join(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0], 'data'))
    filename = os.path.join(prefix, 'atis.pkl')
    train_set, test_set, dicts = cPickle.load(open(filename))
    return train_set, test_set, dicts


def download(origin, destination):
    """
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    """
    print('Downloading data from %s' % origin)
    urllib.urlretrieve(origin, destination)


def get_perf(filename, folder):
    """ run conlleval.pl perl script to obtain precision/recall and F1 score """
    _conlleval = os.path.join(folder, 'conlleval.pl')
    if not os.path.isfile(_conlleval):
        url = 'http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl'
        download(url, _conlleval)
        os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    # 确保安装了perl并添加至PATH中
    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()).encode('utf-8'))
    stdout = stdout.decode('utf-8')
    out = None

    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    # To help debug
    if out is None:
        print(stdout.split('\n'))
    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def conlleval(p, g, w, filename, script_path):
    """
    :param p: predictions
    :param g: groundtruth
    :param w: corresponding words
    :param filename: name of the file where the predictions are written.
        it will be the input of conlleval.pl script for computing the performance
        in terms of precision recall and f1 score
    :param script_path: path to the directory containing the conlleval.pl script
    :return:
    """

    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()
    # 利用perl脚本计算分类后的F1值
    # 多类别时利用one vs others分别计算F1值，最后求平均
    return get_perf(filename, script_path)

###################


def test_atis():
    import pdb

    w2ne, w2la = {}, {}
    train, test, dic = atisfull()
    train, _, test, dic = atisfold(1)

    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

    idx2w = dict((v, k) for k, v in w2idx.iteritems())
    idx2ne = dict((v, k) for k, v in ne2idx.iteritems())
    idx2la = dict((v, k) for k, v in labels2idx.iteritems())

    test_x, test_ne, test_label = test
    train_x, train_ne, train_label = train
    wlength = 35

    for e in ['train', 'test']:
        for sw, se, sl in zip(eval(e + '_x'), eval(e + '_ne'), eval(e + '_label')):
            print 'WORD'.ljust(wlength), 'LABEL'.ljust(wlength), 'TABEL'.ljust(wlength)
            for wx, la, i in zip(sw, sl, se):
                print idx2w[wx].ljust(wlength), idx2la[la].ljust(wlength), idx2ne[i].ljust(wlength)
            print '\n' + '**' * 30 + '\n'
            pdb.set_trace()


def test_context_window():
    x = np.asanyarray([0, 1, 2, 3, 4], dtype=np.int32)
    pprint(context_window(x, 3))
    print
    pprint(context_window(x, 7))

if __name__ == '__main__':
    test_context_window()
