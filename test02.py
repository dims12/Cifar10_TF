# подбираю преобразование для получения картинки в numpy

from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

batch = cifar10readfile(cifar10batches()[0])

for i in range(10000):

    image = batch['data'][i, :]

    image = np.reshape(image, [32, 32, 3], 'F')
    image = np.transpose(image, [1, 0, 2])

    plt.imshow(image)
    plt.show()

pass