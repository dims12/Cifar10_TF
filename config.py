import os
import _pickle as cPickle

def cifar10dir():
    return os.environ['CIFAR10_LOCAL']

def cifar10batchnames():
    return ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

def cifar10testnames():
    return ['test_batch']


def cifar10batches():
    return [ os.path.join( cifar10dir(), name) for name in cifar10batchnames()]

def cifar10tests():
    return [ os.path.join( cifar10dir(), name) for name in cifar10testnames()]

def cifar10readfile(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def cifar10readtest():
    return cifar10readfile(cifar10tests()[0])
