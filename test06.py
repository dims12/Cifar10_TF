# подбираю преобразование кучи картинок
# при произвольной длине пакета

from config import *
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf



x0 = tf.placeholder(tf.uint8, shape=[None, 3072])
x1 = tf.reshape(x0, [-1, 3, 1024])
x2 = tf.transpose(x1, [0, 2, 1])
x3 = tf.reshape(x2, [-1, 32, 32, 3])

batch = cifar10readfile(cifar10batches()[0])

imagessize = 10
images = batch['data'][0:imagessize, :]

sess = tf.InteractiveSession()
images = sess.run(x3, {x0: images})

for i in range(imagessize):
    image = images[i, :, :, :]
    image = np.reshape(image, [32, 32, 3])

    plt.imshow(image)
    plt.show()
#    print(image)

pass