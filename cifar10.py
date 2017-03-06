import _pickle as cPickle
import tensorflow as tf
import random
import numpy as np

from config import *

# def unpickle(file):
#     fo = open(file, 'rb')
#     dict = cPickle.load(fo, encoding='latin1')
#     fo.close()
#     return dict

def batch_reshape(x0):

    x1 = tf.reshape(x0, [-1, 3, 1024])
    x2 = tf.transpose(x1, [0, 2, 1])
    x3 = tf.reshape(x2, [-1, 32, 32, 3])
    return x3

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



#sess = tf.InteractiveSession()
sess = tf.Session()

# to reproduce pathway
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

x = tf.placeholder(tf.float32, shape=[None, 1024*3])
l = tf.placeholder(tf.uint8, shape=[None])

y_ = tf.one_hot(l, 10)

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x_image = batch_reshape(x)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

testsize = 1000
idx = random.sample(range(10000), testsize)

testbatch = cifar10readtest()
testdata = testbatch['data'][idx,:]
testlabels = [testbatch['labels'][k] for k in idx]

# перебираем большие файлы корпуса
batchsize = 1000
for i in range(100):
    bigbatch = cifar10readfile(cifar10batches()[i%5])
    for j in range(20000):
        idx = random.sample(range(10000), batchsize)
        batch = bigbatch['data'][idx, :]
        labels = [bigbatch['labels'][k] for k in idx]
        #labels = np.reshape(labels, [batchsize, 1])
        if j%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch, l: labels, keep_prob: 1.0})
            print("bigbatch %d (%d), step %d, training accuracy %g"%(i, i%5, j, train_accuracy))
            if j%1000 == 0:
                test_accuracy = sess.run(accuracy, feed_dict={x: testdata, l: testlabels, keep_prob: 1.0})
                print("bigbatch %d (%d), step %d, training accuracy %g, test accuracy %g"%(i, i%5, j, train_accuracy, test_accuracy))

        sess.run(train_step, feed_dict={x: batch, l: labels, keep_prob: 0.5})

for i in range(5):
    bigbatch = cifar10readfile(cifar10batches()[i])
    print("bigbatch %d gave accuracy %g"%(i, sess.run(accuracy, feed_dict={x: bigbatch['data'], l: bigbatch['labels'], keep_prob: 1.0})))

print("test gave accuracy %g"%(sess.run(accuracy, feed_dict={x: testdata, l: testlabels, keep_prob: 1.0})))


