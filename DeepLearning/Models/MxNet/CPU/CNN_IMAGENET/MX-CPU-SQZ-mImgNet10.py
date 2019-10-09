from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import logging
import argparse
import time
import mxnet as mx
import mxnet
import math
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
#from multiprocessing import cpu_count

mx.random.seed(1)
PATH = os.environ.get("DATASET_10SET", None)
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
#mx.nd.save('mean_std_224.nd', {"mean_img": mean_img, "std_img": std_img})

train_path = os.path.join(PATH, 'train')
val_path = os.path.join(PATH, 'val')

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(training_transformer),
    batch_size=args.b, shuffle=True, num_workers=4)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(validation_transformer),
    batch_size=args.b, shuffle=False, num_workers=4)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

###########
class FireConv(gluon.nn.HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(FireConv, self).__init__()
        with self.name_scope():
            self.conv = gluon.nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                in_channels=in_channels)
            self.activ = gluon.nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class FireUnit(gluon.nn.HybridBlock):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, residual):
        super(FireUnit, self).__init__()
        self.residual = residual

        with self.name_scope():
            self.squeeze = FireConv(
                in_channels=in_channels,
                out_channels=squeeze_channels,
                kernel_size=1,
                padding=0)
            self.expand1x1 = FireConv(
                in_channels=squeeze_channels,
                out_channels=expand1x1_channels,
                kernel_size=1,
                padding=0)
            self.expand3x3 = FireConv(
                in_channels=squeeze_channels,
                out_channels=expand3x3_channels,
                kernel_size=3,
                padding=1)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = F.concat(y1, y2, dim=1)
        if self.residual:
            out = out + identity
        return out


class SqueezeInitBlock(gluon.nn.HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SqueezeInitBlock, self).__init__()
        with self.name_scope():
            self.conv = gluon.nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=2,
                in_channels=in_channels)
            self.activ = gluon.nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class SqueezeNet(gluon.nn.HybridBlock):
    def __init__(self, channels, residuals, init_block_kernel_size, init_block_channels, in_channels=3, in_size=(224, 224),
                 classes=10):
        super(SqueezeNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = gluon.nn.HybridSequential(prefix="")
            self.features.add(SqueezeInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                kernel_size=init_block_kernel_size))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = gluon.nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    stage.add(gluon.nn.MaxPool2D(
                        pool_size=3,
                        strides=2,
                        ceil_mode=True))
                    for j, out_channels in enumerate(channels_per_stage):
                        expand_channels = out_channels // 2
                        squeeze_channels = out_channels // 8
                        stage.add(FireUnit(
                            in_channels=in_channels,
                            squeeze_channels=squeeze_channels,
                            expand1x1_channels=expand_channels,
                            expand3x3_channels=expand_channels,
                            residual=((residuals is not None) and (residuals[i][j] == 1))))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(gluon.nn.Dropout(rate=0.5))

            self.output = gluon.nn.HybridSequential(prefix="")
            self.output.add(gluon.nn.Conv2D(
                channels=classes,
                kernel_size=1,
                in_channels=in_channels))
            self.output.add(gluon.nn.Activation("relu"))
            self.output.add(gluon.nn.AvgPool2D(
                pool_size=13,
                strides=1))
            self.output.add(gluon.nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

################################## 
channels = [[128, 128, 256], [256, 384, 384, 512], [512]]
residuals = [[0, 1, 0], [1, 0, 1, 0], [1]]
init_block_kernel_size = 7
init_block_channels = 96

net = SqueezeNet(channels=channels, residuals=residuals, init_block_kernel_size=init_block_kernel_size, init_block_channels=init_block_channels)
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
    print('[Epoch %d] training: err-top1=%f err-top5=%f'%(e, err_top1, err_top5))
    print('[Epoch %d] time cost: %f'%(e, time.time()-tic))
