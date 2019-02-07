import os
import urllib.request
import sys
import tensorflow as tf
import numpy as np
import glob

learning_rate = 0.01
dropout_rate = 0.5
classNum = 1000
skip = []
testPath = "testModel"
testImg = []
epochs = 10
batchSize = 4

train_file = 
val_file = 

#usar para dar sequencia.
#https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/datagenerator.py
#https://github.com/fastai/courses/blob/master/deeplearning2/imagenet_process.ipynb

tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)

val_data = ImageDataGenerator(val_file,
                                 mode='inference',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=False)

iterator = Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1], strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha, beta = beta, bias = bias, name = name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding = "SAME", groups = 1):
    """convolution"""
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("b", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)

class alexNet(object):
    def __init__(self, x, keepPro, classNum, skip):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.buildCNN()

    def buildCNN(self):
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEPPRO)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

 
imgMean = np.array([104, 117, 124], np.float)
x = tf.placeholder("float", [1, 227, 227, 3])
y_true = tf.placeholder(tf.float32, shape = [None, 10])
  
model = alexnet.alexNet(x, dropout_rate, classNum, skip)

score = model.fc3
softmax = tf.nn.softmax(score)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = score)) 

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cross_entropy)
  
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        sess.run(training_init_op)
        for i in range(batchSize):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})
            matches = tf.equal(tf.argmax(score, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
