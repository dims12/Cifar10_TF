import os

def cifar10dir():
    return 'V:\\CorporaCopies\\CIFAR-10\\cifar-10-batches-py\\'

def cifar10batchnames():
    return ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']


def cifar10batches():
    return [ os.path.join( cifar10dir(), name) for name in cifar10batchnames()]
