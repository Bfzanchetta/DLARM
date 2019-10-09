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
import sys

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

mx.random.seed(1)
ctx = mx.cpu()

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

train_path = os.path.join('../../10set/', 'train')
val_path = os.path.join('../../10set/', 'val')

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(training_transformer),
    batch_size=args.b, shuffle=True, num_workers=4)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(validation_transformer),
    batch_size=args.b, shuffle=False, num_workers=4)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

##################################

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(
        gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=2, strides=2),
        gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        gluon.nn.MaxPool2D(pool_size=2, strides=2),
        gluon.nn.Flatten(),
        gluon.nn.Dense(128, activation="relu"),
        gluon.nn.Dense(10)
    )


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
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
    print('[Epoch %d] training: err-top1=%f err-top5=%f'%(epochs, err_top1, err_top5))
    print('[Epoch %d] time cost: %f'%(epochs, time.time()-tic))
