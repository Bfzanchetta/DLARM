from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import gluoncv
import numpy as np
import os
import logging
import argparse
import time
import mxnet as mx
import mxnet
import math
from mxnet import nd, autograd, gluon
from mxnet import init
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
#from multiprocessing import cpu_count

mx.random.seed(1)

parser = argparse.ArgumentParser(description="AlexNet for mImgNet10 on MxNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n', type=int, default=15,
                    help='Max number of epochs.')
parser.add_argument('--b', type=int, default=4,
                    help='Batch Size.')
parser.add_argument('--l', type=float, default=0.001,
                    help='Initial Learning Rate.')
parser.add_argument('--s', type=float, default=0.015,
                    help='Smoothing Constant.')
args = parser.parse_args()

ctx = mx.gpu()

num_outputs = 10
jitter_param = 0.4
lighting_param = 0.1

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size=args.b, shuffle=True)
val_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size=args.b, shuffle=False)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
###########
class Residual(gluon.nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = gluon.nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = gluon.nn.BatchNorm()
            self.conv2 = gluon.nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = gluon.nn.BatchNorm()
            if not same_shape:
                self.conv3 = gluon.nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

class ResNet(gluon.nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = gluon.nn.HybridSequential()
            # block 1
            net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1))
            net.add(gluon.nn.BatchNorm())
            net.add(gluon.nn.Activation(activation='relu'))
            # block 2
            for _ in range(3):
                net.add(Residual(channels=32))
            # block 3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=64))
            # block 4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))
            # block 5
            net.add(gluon.nn.AvgPool2D(pool_size=7))
            net.add(gluon.nn.Flatten())
            net.add(gluon.nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out

net = ResNet(10)
##################################
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.l})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

epochs = args.n
smoothing_constant = args.s
moving_loss = 0.

for e in range(epochs):
    tic = time.time()
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        acc_top1.update(label, output)
        acc_top5.update(label, output)
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    err_top1, err_top5 = (1-top1, 1-top5)
    test_accuracy = evaluate_accuracy(val_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)

    file_name = "mx_gpu_res_mnist.params"
    net.save_parameters(file_name)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
    print('[Epoch %d] training: err-top1=%f err-top5=%f'%(e, err_top1, err_top5))
    print('[Epoch %d] time cost: %f'%(e, time.time()-tic))

