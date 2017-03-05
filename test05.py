# подбираю преобразование кучи картинок

from cifar10 import *
from config import *
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


batchsize = 10
x0 = tf.placeholder(tf.uint8, shape=[batchsize, 3072])
x1 = tf.reshape(x0, [batchsize, 3, 1024])
x2 = tf.transpose(x1, [0, 2, 1])
x3 = tf.reshape(x2, [batchsize, 32, 32, 3])

batch = unpickle(cifar10batches()[0])

images = batch['data'][0:batchsize, :]

sess = tf.InteractiveSession()
images = sess.run(x3, {x0: images})

for i in range(batchsize):
    image = images[i, :, :, :]
    image = np.reshape(image, [32, 32, 3])

    plt.imshow(image)
    plt.show()
#    print(image)

pass