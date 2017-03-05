# подбираю преобразование для картинки в tensorflow

from cifar10 import *
from config import *
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
sess = tf.InteractiveSession()


x0 = tf.placeholder(tf.uint8, shape=[1, 3072])
x1 = tf.reshape(x0, [3, 1024])
x2 = tf.transpose(x1)
x3 = tf.reshape(x2, [32, 32, 3])

batch = unpickle(cifar10batches()[0])

for i in range(10000):



    image = np.reshape(batch['data'][i, :], [1, 3072])

    image = sess.run(x3, {x0: image})

    plt.imshow(image)
    plt.show()
    # print(image)

