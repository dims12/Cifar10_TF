# подбираю преобразование для кучи картинок в numpy

from cifar10 import *
from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

bigbatch = unpickle(cifar10batches()[0])

imagessize = 10
images = bigbatch['data'][0:imagessize]

print("initial shape: ", images)

images = np.reshape(images, [imagessize*3, 1024])
images = np.reshape(images, [imagessize, 3, 1024])
images = np.transpose(images, [0, 2, 1])
images = np.reshape(images, [imagessize, 32, 32, 3])


for i in range(imagessize):
    image = images[i, :, :, :]
    plt.imshow(image)
    plt.show()


pass