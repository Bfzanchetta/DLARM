from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import gluoncv
import numpy as np
import os
import time
import mxnet as mx
import mxnet
import math
from mxnet import nd, autograd, gluon
from mxnet import init
#from gluoncv.data import ImageNet
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
#from multiprocessing import cpu_count
#https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/gluon_from_experiment_to_deployment.html
######
import sys
sys.path.append('..')
import utils
######


mx.random.seed(1)
# ctx = mx.gpu()
ctx = mx.gpu()

batch_size = 4
num_outputs = 10

jitter_param = 0.4
lighting_param = 0.1

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

training_transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

validation_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

mean_img = mx.nd.stack(*[mx.nd.full((224, 224), m) for m in mean])
std_img = mx.nd.stack(*[mx.nd.full((224, 224), s) for s in std])
mx.nd.save('mean_std_224.nd', {"mean_img": mean_img, "std_img": std_img})

train_path = os.path.join('/home/dlarm/10set/', 'train')
val_path = os.path.join('/home/dlarm/10set/', 'val')

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(training_transformer),
    batch_size=4, shuffle=True, num_workers=4)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(validation_transformer),
    batch_size=4, shuffle=False, num_workers=4)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

##################################

class Residual(gluon.nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = gluon.nn.Conv2D(channels, kernel_size=3, padding=1,
                              strides=strides)
        self.bn1 = gluon.nn.BatchNorm()
        self.conv2 = gluon.nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = gluon.nn.BatchNorm()
        if not same_shape:
            self.conv3 = gluon.nn.Conv2D(channels, kernel_size=1,
                                  strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)

blk = Residual(3)
blk.initialize()

x = nd.random.uniform(shape=(4, 3, 6, 6))
blk(x).shape

blk2 = Residual(8, same_shape=False)
blk2.initialize()
blk2(x).shape

class ResNet(gluon.nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = gluon.nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = gluon.nn.Sequential()
            b2.add(
                gluon.nn.MaxPool2D(pool_size=3, strides=2),
                Residual(64),
                Residual(64)
            )
            # block 3
            b3 = gluon.nn.Sequential()
            b3.add(
                Residual(128, same_shape=False),
                Residual(128)
            )
            # block 4
            b4 = gluon.nn.Sequential()
            b4.add(
                Residual(256, same_shape=False),
                Residual(256)
            )
            # block 5
            b5 = gluon.nn.Sequential()
            b5.add(
                Residual(512, same_shape=False),
                Residual(512)
            )
            # block 6
            b6 = gluon.nn.Sequential()
            b6.add(
                gluon.nn.AvgPool2D(pool_size=3),
                gluon.nn.Dense(num_classes)
            )
            # chain all blocks together
            self.net = gluon.nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out

net = ResNet(10, verbose=True)
net.initialize()

x = nd.random.uniform(shape=(4, 3, 96, 96))
y = net(x)

##################################

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

epochs = 10
smoothing_constant = .015
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
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
    print('[Epoch %d] training: err-top1=%f err-top5=%f'%(epochs, err_top1, err_top5))
    print('[Epoch %d] time cost: %f'%(epochs, time.time()-tic))
